import time
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil
import GPUtil
from tensorflow.keras import layers, models, losses, optimizers, metrics as keras_metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from tensorflow_model_optimization.sparsity import keras as sparsity
import os

# For inference-only timing later
proc = psutil.Process()

os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=./hlo_dumps '
    '--xla_dump_fusion_visualization '
    '--xla_hlo_profile'
)
tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print("Available CPU devices:", cpus)
print("Available GPU devices:", gpus)

# ==============================
# 1. Preprocessing
# ==============================
df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\wustl_iiot_1.csv',
    dtype=str, na_values=['?', '-'], keep_default_na=True, low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.drop(columns=['Timestamp','LastTime','SrcIP','DstIP','Target','sIpId','dIpId'

                ,'DIntPkt', 'TotalBytes', 'DstRate', 'Loss', 'Protocol', 'DstJitAct', 'TotAppByte', 'TcpRtt', 'SynAck', 'IdleTime', 'TotalPkts', 'SrcPkts', 'SrcBytes', 'Duration'
              ,'DstLoad', 'sDSb', 'sTos', 'DstPkts', 'DstJitter'
             ,'SrcJitAct', 'Sum', 'RunTime', 'Max', 'Min', 'Mean', 'SIntPkt', 'SrcJitter', 'SAppBytes'
                 ], errors='ignore', inplace=True)

X = df.drop(columns=['Traffic']).apply(pd.to_numeric, errors='coerce')
y = df['Traffic'].fillna('missing').astype(str)

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

# Encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_enc, test_size=0.3, stratify=y_enc, random_state=42
)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape
seq_len = X_train.shape[1]
feat_dim = 1
X_train = X_train.reshape(-1, seq_len, feat_dim)
X_test  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# ==============================
# 2. Teacher Model
# ==============================
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

    scores = layers.Dense(1, activation='tanh')(x)
    weights = layers.Softmax(axis=1)(scores)
    weighted = layers.Multiply()([x, weights])
    ctx = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(weighted)

    out = layers.Dense(num_classes, activation='softmax')(ctx)
    model = models.Model(inp, out, name='Teacher')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

teacher = build_teacher()

# ==============================
# 3. Train Teacher
# ==============================
t0 = time.time()
teacher.fit(X_train, y_train, validation_split=0.1, epochs=1, batch_size=256, verbose=1)
teacher_time = time.time() - t0
print(f"\nTeacher training time: {teacher_time:.2f}s")
te_loss, te_acc = teacher.evaluate(X_test, y_test, verbose=0)
print(f"Teacher eval loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# ==============================
# 4. Soft Labels
# ==============================
T = 10.0
train_logits = teacher.predict(X_train, batch_size=512)
soft_train = tf.nn.softmax(train_logits / T)
test_logits  = teacher.predict(X_test, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T)

# ==============================
# 5. Dataset Pipeline
# ==============================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, soft_train)) \
    .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test, soft_test)) \
    .batch(256).prefetch(tf.data.AUTOTUNE)

# ==============================
# 6. Student Model
# ==============================
def build_student():
    inp = layers.Input(shape=(seq_len, feat_dim))
    x = layers.GRU(64, implementation=2)(inp)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='Student_GRU')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

student = build_student()

# ==============================
# 7. Train Student 
# ==============================
print("\n=== Standalone Student Training ===")
wall_before = time.time()
cpu_before = proc.cpu_times().user + proc.cpu_times().system
ram_before = proc.memory_info().rss

student.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=1,
    batch_size=256,
    verbose=1
)

wall_after = time.time()
cpu_after = proc.cpu_times().user + proc.cpu_times().system
ram_after = proc.memory_info().rss

print(f"Standalone student training wall time: {wall_after - wall_before:.2f}s")
print(f"Standalone student training CPU time: {cpu_after - cpu_before:.2f}s")
print(f"Standalone student training RAM Δ:      {(ram_after - ram_before)/1024**2:.2f} MB")

st_loss, st_acc = student.evaluate(X_test, y_test, verbose=0)
print(f"Standalone student eval loss: {st_loss:.4f}, acc: {st_acc:.4f}")

# ==============================
# 8. Distiller 
# ==============================
class Distiller(models.Model):
    def __init__(self, student, temp=10.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.temp = temp
        self.alpha = alpha

    def compile(self,
                optimizer=None,
                student_loss=None,
                distill_loss=None,
                metrics_list=None,
                *args, **kwargs):
        super().compile(
            optimizer=optimizer,
            loss=student_loss,
            metrics=metrics_list or [keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
            *args, **kwargs
        )
        self.s_loss = student_loss
        self.d_loss = distill_loss

    @tf.function
    def train_step(self, data):
        x, y, t = data
        with tf.GradientTape() as tape:
            preds = self.student(x, training=True)
            sl = self.s_loss(y, preds)
            ps = tf.nn.softmax(preds / self.temp)
            dl = self.d_loss(t, ps) * (self.temp ** 2)
            loss = self.alpha * sl + (1 - self.alpha) * dl
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

# ==============================
# 9. Train Distiller on GPU
# ==============================
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    distiller = Distiller(student, temp=T, alpha=0.5)
    distiller.compile(
        optimizer=optimizers.Adam(),
        student_loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        distill_loss=losses.KLDivergence()
    )
    d0 = time.time()
    distiller.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    print(f"\nDistillation training time: {time.time() - d0:.2f}s")

print("\n=== Post-Distillation Student Evaluation ===")
wall_before = time.time()
cpu_before = proc.cpu_times().user + proc.cpu_times().system
ram_before = proc.memory_info().rss

st_loss_kd, st_acc_kd = student.evaluate(X_test, y_test, verbose=0)

wall_after = time.time()
cpu_after = proc.cpu_times().user + proc.cpu_times().system
ram_after = proc.memory_info().rss

print(f"Post-distillation student eval wall time: {wall_after - wall_before:.2f}s")
print(f"Post-distillation student eval CPU time:  {cpu_after - cpu_before:.2f}s")
print(f"Post-distillation student eval RAM Δ:       {(ram_after - ram_before)/1024**2:.4f} MB")
print(f"Post-distillation student loss: {st_loss_kd:.4f}, acc: {st_acc_kd:.4f}")

# ==============================
# 11. Visualization
# ==============================
y_pred = student.predict(X_test, batch_size=512)
y_labels = np.argmax(y_pred, axis=1)

report_dict = classification_report(
    y_test, y_labels, target_names=le.classes_, output_dict=True
)
df_report = pd.DataFrame(report_dict).T
for col in ['precision', 'recall', 'f1-score']:
    df_report[col] = (df_report[col] * 100).map(lambda x: f"{x:.2f}")
df_report['support'] = df_report['support'].astype(int)
print("\nFormatted Classification Report (percentages):")
print(df_report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.tight_layout()
plt.show()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i],
             label=f"{le.classes_[i]} (AUC = {roc_auc[i]:.2f})")
# plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

mem_inf0 = proc.memory_info().rss / (1024**2)
start_inf = time.time()
y_prob_inf = student.predict(X_test, batch_size=256, verbose=0)
inf_time = time.time() - start_inf
mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")
