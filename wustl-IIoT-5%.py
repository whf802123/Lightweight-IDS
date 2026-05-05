import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import psutil
import pandas as pd
import numpy as np
import tensorflow as tf
import GPUtil
from tensorflow.keras import layers, models, losses, optimizers, metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print("Available CPU devices:", cpus)
print("Available GPU devices:", gpus)

# Process object for RAM and CPU stats
proc = psutil.Process(os.getpid())

# Load & preprocess train/test sets separately
train_df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\wustl_iiot_1_5pct_train.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
test_df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\wustl_iiot_1_5%-test.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)

def preprocess(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop(columns=['Timestamp', 'LastTime', 'SrcIP', 'DstIP', 'Target', 'sIpId', 'dIpId'

                        ,'DIntPkt', 'TotalBytes', 'DstRate', 'Loss', 'Protocol', 'DstJitAct', 'TotAppByte', 'TcpRtt', 'SynAck', 'IdleTime', 'TotalPkts', 'SrcPkts', 'SrcBytes', 'Duration'
                        ,'DstLoad', 'sDSb', 'sTos', 'DstPkts', 'DstJitter'
                       ,'SrcJitAct', 'Sum', 'RunTime', 'Max', 'Min', 'Mean', 'SIntPkt', 'SrcJitter', 'SAppBytes'
                     ], errors='ignore', inplace=True)

    X = df.drop(columns=['Traffic']).apply(pd.to_numeric, errors='coerce')
    y = df['Traffic'].fillna('missing').astype(str)
    return X, y

X_train_raw, y_train_raw = preprocess(train_df)
X_test_raw,  y_test_raw  = preprocess(test_df)

# Median imputation
imputer    = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp  = imputer.transform(X_test_raw)

# Label encoding based on train labels
le           = LabelEncoder()
y_train_enc  = le.fit_transform(y_train_raw)
y_test_enc   = le.transform(y_test_raw)
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_imp)
X_test  = scaler.transform(X_test_imp)

# reshape for sequence models
seq_len    = X_train.shape[1]
feat_dim   = 1
X_train    = X_train.reshape(-1, seq_len, feat_dim)
X_test     = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# 2. Build Teacher model 
def build_teacher():
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    def tcn_block(x, rate):
        conv = layers.Conv1D(64, 3, padding='causal',
                             dilation_rate=rate, activation='relu')(x)
        conv = layers.BatchNormalization()(conv)
        return layers.Add()([x, conv])
    for rate in [1, 2]:
        x = tcn_block(x, rate)
    scores   = layers.Dense(1, activation='tanh')(x)
    weights  = layers.Softmax(axis=1)(scores)
    weighted = layers.Multiply()([x, weights])
    ctx      = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(weighted)
    out      = layers.Dense(num_classes, activation='softmax')(ctx)
    model    = models.Model(inp, out, name='Teacher')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    return model

teacher = build_teacher()

# Train Teacher
t0 = time.time()
teacher.fit(X_train, y_train_enc, validation_split=0.1,
            epochs=1, batch_size=256, verbose=1)
print(f"Teacher training time: {time.time() - t0:.2f}s")
te_loss, te_acc = teacher.evaluate(X_test, y_test_enc, verbose=0)
print(f"Teacher eval   loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# Soft labels
T            = 10.0
train_logits = teacher.predict(X_train, batch_size=512)
soft_train   = tf.nn.softmax(train_logits / T)
test_logits  = teacher.predict(X_test, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T)

# Build dataset pipeline 
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_enc, soft_train)) \
               .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_test,   y_test_enc,  soft_test)) \
               .batch(256).prefetch(tf.data.AUTOTUNE)

# Build Student model 
def build_student():
    inp = layers.Input(shape=(seq_len, feat_dim))
    x   = layers.GRU(64, implementation=2)(inp)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='Student_GRU')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    return model

student = build_student()

# Student training & stats
print("\n=== Standalone Student Training ===")
wall_before = time.time()
cpu_before  = proc.cpu_times().user + proc.cpu_times().system
ram_before  = proc.memory_info().rss

student.fit(X_train, y_train_enc, validation_split=0.1,
            epochs=1, batch_size=256, verbose=1)

print(f"Wall time (train): {time.time() - wall_before:.2f}s")
print(f"CPU  time (train): {proc.cpu_times().user + proc.cpu_times().system - cpu_before:.2f}s")
print(f"RAM Δ  (train):    {(proc.memory_info().rss - ram_before)/1024**2:.2f} MB")

st_loss, st_acc = student.evaluate(X_test, y_test_enc, verbose=0)
print(f"Standalone student eval loss: {st_loss:.4f}, acc: {st_acc:.4f}")

# Distiller
class Distiller(models.Model):
    def __init__(self, student, temp=10.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.temp    = temp
        self.alpha   = alpha

    def compile(self, optimizer, student_loss, distill_loss):
        super().compile(
            optimizer=optimizer,
            loss=student_loss,
            metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
        )
        self.s_loss = student_loss
        self.d_loss = distill_loss

    @tf.function
    def train_step(self, data):
        x, y, t = data
        with tf.GradientTape() as tape:
            preds = self.student(x, training=True)
            sl    = self.s_loss(y, preds)
            ps    = tf.nn.softmax(preds / self.temp)
            dl    = self.d_loss(t, ps) * (self.temp**2)
            loss  = self.alpha * sl + (1 - self.alpha) * dl
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y, t = data
        preds = self.student(x, training=False)
        self.compiled_metrics.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

# Train Distiller 
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    distiller = Distiller(student, temp=T, alpha=0.5)
    distiller.compile(
        optimizer=optimizers.Adam(),
        student_loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        distill_loss=losses.KLDivergence()
    )
    d0 = time.time()
    distiller.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    print(f"Distillation training time: {time.time() - d0:.2f}s")

# Post-distillation evaluation & stats
print("\n=== Post-Distillation Student Evaluation ===")
wall_before = time.time()
cpu_before  = proc.cpu_times().user + proc.cpu_times().system
ram_before  = proc.memory_info().rss

st_loss_kd, st_acc_kd = student.evaluate(X_test, y_test_enc, verbose=0)

print(f"Post-distill eval wall time: {time.time() - wall_before:.2f}s")
print(f"Post-distill eval CPU time:  {proc.cpu_times().user + proc.cpu_times().system - cpu_before:.2f}s")
print(f"Post-distill eval RAM Δ:     {(proc.memory_info().rss - ram_before)/1024**2:.2f} MB")
print(f"Post-distillation student loss: {st_loss_kd:.4f}, acc: {st_acc_kd:.4f}")

# Classification report
y_pred = student.predict(X_test, batch_size=512)
report_dict = classification_report(
    y_test_enc, np.argmax(y_pred, axis=1),
    target_names=le.classes_, output_dict=True
)
df_report = pd.DataFrame(report_dict).T
for col in ['precision', 'recall', 'f1-score']:
    df_report[col] = (df_report[col] * 100).map(lambda x: f"{x:.2f}")
df_report['support'] = df_report['support'].astype(int)
print("\nFormatted Classification Report (percentages):")
print(df_report)

# Inference Time and Memory Usage
mem_inf0 = proc.memory_info().rss / (1024**2)
start_inf = time.time()
y_prob_inf = student.predict(X_test, batch_size=256, verbose=0)
inf_time = time.time() - start_inf
mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")

