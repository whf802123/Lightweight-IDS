
'''
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
import os

# For inference-only timing later
proc = psutil.Process()

os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=./hlo_dumps '
    '--xla_dump_fusion_visualization '
    '--xla_hlo_profile'
)
tf.config.optimizer.set_jit(True)

# ==============================
# 0. List available devices
# ==============================
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print("Available CPU devices:", cpus)
print("Available GPU devices:", gpus)

# ==============================
# 1. Load & preprocess data
# ==============================
df = pd.read_csv(
    r'C:\\Users\\Administrator\\Desktop\\wustl_iiot_1.csv',
    dtype=str, na_values=['?', '-'], keep_default_na=True, low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.drop(columns=[
    'Timestamp','LastTime','SrcIP','DstIP','Target','sIpId','dIpId',
    'DIntPkt', 'TotalBytes', 'DstRate', 'Loss', 'Protocol', 'DstJitAct',
    'TotAppByte', 'TcpRtt', 'SynAck', 'IdleTime', 'TotalPkts', 'SrcPkts',
    'SrcBytes', 'Duration','DstLoad', 'sDSb', 'sTos', 'DstPkts',
    'DstJitter','SrcJitAct', 'Sum', 'RunTime', 'Max', 'Min', 'Mean',
    'SIntPkt', 'SrcJitter', 'SAppBytes'
], errors='ignore', inplace=True)

X = df.drop(columns=['Traffic']).apply(pd.to_numeric, errors='coerce')
y = df['Traffic'].fillna('missing').astype(str)

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

# Label encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_enc, test_size=0.3, stratify=y_enc, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape to (batch, seq_len, feat_dim)
seq_len = X_train.shape[1]
feat_dim = 1
X_train = X_train.reshape(-1, seq_len, feat_dim)
X_test  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# ==============================
# 2. Define LSH self-attention and GCN layers
# ==============================
class LSHSelfAttention(layers.Layer):
    def __init__(self, num_hashes, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_hashes = num_hashes
        self.key_dim = key_dim

    def build(self, input_shape):
        self.q_dense = layers.Dense(self.key_dim, use_bias=False)
        self.k_dense = layers.Dense(self.key_dim, use_bias=False)
        self.v_dense = layers.Dense(self.key_dim, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        # TODO: replace with true LSH bucketing
        return layers.Attention()([q, k, v])

class GraphConv(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True)
        super().build(input_shape)

    def call(self, x, adj=None):
        return tf.matmul(x, self.w)

# ==============================
# 3. Build Teacher model: IoT-FKGD
# ==============================
def build_teacher_iotfkgd():
    inp = layers.Input((seq_len, feat_dim))

    # Multi-scale dilated convolutions
    x1 = layers.Conv1D(128, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    x2 = layers.Conv1D(128, 3, padding='causal', dilation_rate=2, activation='relu')(inp)
    x4 = layers.Conv1D(128, 3, padding='causal', dilation_rate=4, activation='relu')(inp)
    x = layers.Concatenate()([x1, x2, x4])

    # LSH-based self-attention
    x = LSHSelfAttention(num_hashes=8, key_dim=256)(x)

    # GCN hidden 2048
    x = GraphConv(2048)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global pooling + MLP head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(2048, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out, name='Teacher_IoT_FKGD')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

teacher = build_teacher_iotfkgd()

# ==============================
# 4. Train Teacher
# ==============================
t0 = time.time()
teacher.fit(X_train, y_train, validation_split=0.1, epochs=1, batch_size=256, verbose=1)
teacher_time = time.time() - t0
print(f"\nTeacher training time: {teacher_time:.2f}s")
te_loss, te_acc = teacher.evaluate(X_test, y_test, verbose=0)
print(f"Teacher eval loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# ==============================
# 5. Pre-compute soft labels
# ==============================
T = 10.0
train_logits = teacher.predict(X_train, batch_size=512)
soft_train = tf.nn.softmax(train_logits / T)
test_logits  = teacher.predict(X_test, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T)

# ==============================
# 6. Build Dataset pipeline
# ==============================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, soft_train)) \
    .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test, soft_test)) \
    .batch(256).prefetch(tf.data.AUTOTUNE)

# ==============================
# 7. Build Student model: IoT-FKGD (lightweight)
# ==============================
def build_student_iotfkgd():
    inp = layers.Input((seq_len, feat_dim))

    # Multi-scale dilated convolutions
    x1 = layers.Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    x2 = layers.Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(inp)
    x4 = layers.Conv1D(64, 3, padding='causal', dilation_rate=4, activation='relu')(inp)
    x = layers.Concatenate()([x1, x2, x4])

    # LSH-based self-attention (4 hashes)
    x = LSHSelfAttention(num_hashes=4, key_dim=128)(x)

    # GCN hidden 512
    x = GraphConv(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global pooling + MLP head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out, name='Student_IoT_FKGD')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

student = build_student_iotfkgd()

# ==============================
# 8. Standalone Student training & resource profiling
# ==============================
print("\n=== Standalone Student Training ===")
wall_before = time.time()
cpu_before = proc.cpu_times().user + proc.cpu_times().system
ram_before = proc.memory_info().rss

student.fit(X_train, y_train, validation_split=0.1, epochs=1, batch_size=256, verbose=1)

wall_after = time.time()
cpu_after = proc.cpu_times().user + proc.cpu_times().system
ram_after = proc.memory_info().rss

print(f"Standalone student training wall time: {wall_after - wall_before:.2f}s")
print(f"Standalone student training CPU time: {cpu_after - cpu_before:.2f}s")
print(f"Standalone student training RAM Δ:      {(ram_after - ram_before)/1024**2:.2f} MB")

st_loss, st_acc = student.evaluate(X_test, y_test, verbose=0)
print(f"Standalone student eval loss: {st_loss:.4f}, acc: {st_acc:.4f}")

# ==============================
# 9. Define Distiller (fixed compile signature)
# ==============================
class Distiller(models.Model):
    def __init__(self, student, temp=10.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.temp = temp
        self.alpha = alpha

    def compile(self, optimizer=None, student_loss=None, distill_loss=None, metrics_list=None, *args, **kwargs):
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
# 10. Distillation training
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

# ==============================
# 11. Post-distillation evaluation & resource profiling
# ==============================
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
# 12. Output classification report & visualization
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

cm = confusion_matrix(y_test, y_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.tight_layout()
plt.show()

y_test_bin = label_binarize(y_test, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{le.classes_[i]} (AUC = {roc_auc[i]:.2f})")
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

# ==============================
# 13. Pure inference time & memory
# ==============================
mem_inf0 = proc.memory_info().rss / (1024**2)
start_inf = time.time()
_ = student.predict(X_test, batch_size=256, verbose=0)
inf_time = time.time() - start_inf
mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")
'''


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

# ==============================
# 0. List available devices
# ==============================
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print("Available CPU devices:", cpus)
print("Available GPU devices:", gpus)

# Process object for RAM and CPU stats
proc = psutil.Process(os.getpid())

# ==============================
# 1. Load & preprocess train/test sets separately
# ==============================
train_df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\wustl_iiot_1_10pct_train.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
test_df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\wustl_iiot_10%_test.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)

def preprocess(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop(columns=[
        'Timestamp', 'LastTime', 'SrcIP', 'DstIP', 'Target', 'sIpId', 'dIpId',
        'DIntPkt','TotalBytes','DstRate','Loss','Protocol','DstJitAct','TotAppByte',
        'TcpRtt','SynAck','IdleTime','TotalPkts','SrcPkts','SrcBytes','Duration',
        'DstLoad','sDSb','sTos','DstPkts','DstJitter','SrcJitAct','Sum','RunTime',
        'Max','Min','Mean','SIntPkt','SrcJitter','SAppBytes'
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
seq_len     = X_train.shape[1]
feat_dim    = 1
X_train     = X_train.reshape(-1, seq_len, feat_dim)
X_test      = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# ==============================
# 2. Define IoT-FKGD Teacher & Student architectures
# ==============================
class LSHSelfAttention(layers.Layer):
    def __init__(self, num_hashes, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_hashes = num_hashes
        self.key_dim = key_dim

    def build(self, input_shape):
        self.q_dense = layers.Dense(self.key_dim, use_bias=False)
        self.k_dense = layers.Dense(self.key_dim, use_bias=False)
        self.v_dense = layers.Dense(self.key_dim, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        # TODO: insert true LSH bucketing here
        return layers.Attention()([q, k, v])

class GraphConv(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True)
        super().build(input_shape)

    def call(self, x, adj=None):
        return tf.matmul(x, self.w)

def build_teacher_iotfkgd():
    inp = layers.Input((seq_len, feat_dim))
    # multi-scale dilated convs
    x1 = layers.Conv1D(128, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    x2 = layers.Conv1D(128, 3, padding='causal', dilation_rate=2, activation='relu')(inp)
    x4 = layers.Conv1D(128, 3, padding='causal', dilation_rate=4, activation='relu')(inp)
    x  = layers.Concatenate()([x1, x2, x4])
    # LSH-based self-attention (8 hashes)
    x  = LSHSelfAttention(num_hashes=8, key_dim=256)(x)
    # GCN hidden 2048
    x  = GraphConv(2048)(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.ReLU()(x)
    # global pooling + MLP head
    x  = layers.GlobalAveragePooling1D()(x)
    x  = layers.Dense(2048, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out, name='Teacher_IoT_FKGD')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    return model

def build_student_iotfkgd():
    inp = layers.Input((seq_len, feat_dim))
    # multi-scale dilated convs
    x1 = layers.Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    x2 = layers.Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(inp)
    x4 = layers.Conv1D(64, 3, padding='causal', dilation_rate=4, activation='relu')(inp)
    x  = layers.Concatenate()([x1, x2, x4])
    # LSH-based self-attention (4 hashes)
    x  = LSHSelfAttention(num_hashes=4, key_dim=128)(x)
    # GCN hidden 512
    x  = GraphConv(512)(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.ReLU()(x)
    # global pooling + MLP head
    x  = layers.GlobalAveragePooling1D()(x)
    x  = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out, name='Student_IoT_FKGD')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    return model

# Instantiate models
teacher = build_teacher_iotfkgd()
student = build_student_iotfkgd()

# ==============================
# 3. Train Teacher
# ==============================
t0 = time.time()
teacher.fit(X_train, y_train_enc, validation_split=0.1,
            epochs=1, batch_size=256, verbose=1)
print(f"Teacher training time: {time.time() - t0:.2f}s")
te_loss, te_acc = teacher.evaluate(X_test, y_test_enc, verbose=0)
print(f"Teacher eval   loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# ==============================
# 4. Precompute soft labels
# ==============================
T            = 10.0
train_logits = teacher.predict(X_train, batch_size=512)
soft_train   = tf.nn.softmax(train_logits / T)
test_logits  = teacher.predict(X_test,  batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T)

# ==============================
# 5. Build dataset pipeline
# ==============================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_enc, soft_train)) \
               .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_test,  y_test_enc,  soft_test)) \
               .batch(256).prefetch(tf.data.AUTOTUNE)

# ==============================
# 6. Standalone Student training & stats
# ==============================
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

# ==============================
# 7. Define Distiller
# ==============================
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

# ==============================
# 8. Train Distiller
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
    print(f"Distillation training time: {time.time() - d0:.2f}s")

# ==============================
# 9. Post-distillation evaluation & stats
# ==============================
print("\n=== Post-Distillation Student Evaluation ===")
wall_before = time.time()
cpu_before  = proc.cpu_times().user + proc.cpu_times().system
ram_before  = proc.memory_info().rss

st_loss_kd, st_acc_kd = student.evaluate(X_test, y_test_enc, verbose=0)

print(f"Post-distill eval wall time: {time.time() - wall_before:.2f}s")
print(f"Post-distill CPU time:      {proc.cpu_times().user + proc.cpu_times().system - cpu_before:.2f}s")
print(f"Post-distill RAM Δ:         {(proc.memory_info().rss - ram_before)/1024**2:.2f} MB")
print(f"Post-distillation loss:     {st_loss_kd:.4f}, acc: {st_acc_kd:.4f}")

# ==============================
# 10. Formatted classification report
# ==============================
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

# ==============================
# 11. Inference time & memory
# ==============================
mem_inf0 = proc.memory_info().rss / (1024**2)
start_inf = time.time()
_ = student.predict(X_test, batch_size=256, verbose=0)
inf_time = time.time() - start_inf
mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")


