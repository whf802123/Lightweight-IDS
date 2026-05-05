import time
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil
import GPUtil
from tensorflow.keras import layers, models, optimizers, losses, metrics as keras_metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from tensorflow_model_optimization.sparsity import keras as sparsity
import os

'''
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

df = pd.read_csv(
    r'C:\\Users\\Administrator\\Desktop\\wustl_iiot_1.csv',
    dtype=str, na_values=['?', '-'], keep_default_na=True, low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.drop(columns=['Timestamp','LastTime','SrcIP','DstIP','Target','sIpId','dIpId'

              #  ,'DIntPkt', 'TotalBytes', 'DstRate', 'Loss', 'Protocol', 'DstJitAct', 'TotAppByte', 'TcpRtt', 'SynAck', 'IdleTime', 'TotalPkts', 'SrcPkts', 'SrcBytes', 'Duration'
             # ,'DstLoad', 'sDSb', 'sTos', 'DstPkts', 'DstJitter'
            # ,'SrcJitAct', 'Sum', 'RunTime', 'Max', 'Min', 'Mean', 'SIntPkt', 'SrcJitter', 'SAppBytes'
                 ], errors='ignore', inplace=True)

X = df.drop(columns=['Traffic']).apply(pd.to_numeric, errors='coerce')
y = df['Traffic'].fillna('missing').astype(str)

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

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

# Reshape (batch, seq_len, feat_dim)
seq_len = X_train.shape[1]
feat_dim = 1
X_train = X_train.reshape(-1, seq_len, feat_dim)
X_test  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# Teacher Model 
def build_teacher():
    inp = layers.Input(shape=(seq_len, feat_dim), name='teacher_input')
    # Two standard 1D convolutions
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv1')(inp)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False),
        name='bilstm'
    )(x)
    # Classification head
    out = layers.Dense(
        num_classes,
        activation='softmax',
        name='teacher_output'
    )(x)

    model = models.Model(inputs=inp, outputs=out, name='Teacher_BiLSTM')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model


teacher = build_teacher()

# Train Teacher
t0 = time.time()
teacher.fit(X_train, y_train, validation_split=0.1, epochs=1, batch_size=256, verbose=1)
teacher_time = time.time() - t0
print(f"\nTeacher training time: {teacher_time:.2f}s")
te_loss, te_acc = teacher.evaluate(X_test, y_test, verbose=0)
print(f"Teacher eval loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# Soft Labels
T = 10.0
train_logits = teacher.predict(X_train, batch_size=512)
soft_train = tf.nn.softmax(train_logits / T)
test_logits  = teacher.predict(X_test, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T)

# Dataset Pipeline
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, soft_train)) \
    .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test, soft_test)) \
    .batch(256).prefetch(tf.data.AUTOTUNE)

# Student Model
def build_student():
    inp = layers.Input(shape=(seq_len, feat_dim), name='student_input')
    # Depthwise separable convolutions
    x = layers.SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='sep_conv1'
    )(inp)
    x = layers.SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='sep_conv2'
    )(x)
    # Compact bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False),
        name='student_bilstm'
    )(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    # Classification head
    out = layers.Dense(
        num_classes,
        activation='softmax',
        name='student_output'
    )(x)

    model = models.Model(inputs=inp, outputs=out, name='Student_CompactBiLSTM')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model
student = build_student()

# Train Student Model
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

# Distiller 
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
    print(f"\nDistillation training time: {time.time() - d0:.2f}s")

# Evaluation
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

# Visualization
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
    r'C:\\Users\\Administrator\\Desktop\\wustl_iiot_1_10pct_train.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
test_df = pd.read_csv(
    r'C:\\Users\\Administrator\\Desktop\\wustl_iiot_10%_test.csv',
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

# Build Teacher model 
def build_teacher():
    inp = layers.Input(shape=(seq_len, feat_dim), name='teacher_input')
    # Two standard 1D convolutions
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv1')(inp)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False),
        name='bilstm'
    )(x)
    # Classification head
    out = layers.Dense(
        num_classes,
        activation='softmax',
        name='teacher_output'
    )(x)

    model = models.Model(inputs=inp, outputs=out, name='Teacher_BiLSTM')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
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

# Precompute soft labels
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
    inp = layers.Input(shape=(seq_len, feat_dim), name='student_input')
    # Depthwise separable convolutions
    x = layers.SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='sep_conv1'
    )(inp)
    x = layers.SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='sep_conv2'
    )(x)
    # Compact bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False),
        name='student_bilstm'
    )(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    # Classification head
    out = layers.Dense(
        num_classes,
        activation='softmax',
        name='student_output'
    )(x)

    model = models.Model(inputs=inp, outputs=out, name='Student_CompactBiLSTM')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
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

# 10. Post-distillation evaluation & stats
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

# Inference time & RAM Usage
mem_inf0 = proc.memory_info().rss / (1024**2)
start_inf = time.time()
y_prob_inf = student.predict(X_test, batch_size=256, verbose=0)
inf_time = time.time() - start_inf
mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")

'''

import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, metrics

proc = psutil.Process(os.getpid())

df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\X-IIoTID dataset.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

bool_cols = [
    'is_syn_only','Is_SYN_ACK','is_pure_ack','is_with_payload',
    'FIN or RST','Bad_checksum','is_SYN_with_RST','anomaly_alert'
]
for c in bool_cols:
    if c in df.columns:
        df[c] = df[c].map({'TRUE':1, 'FALSE':0})

drop_cols = ['Date','Timestamp','SrcIP','DstIP','class1','class3'

           , 'Std_nice_time', 'DstPkts', 'Scr_ip_bytes', 'TotalPkts, Std_ldavg_1', 'Std_kbmemused', 'SrcPkts', 'Std_wtps', 'Avg_iowait_time', 'Std_iowait_time', 'missed_bytes', 'Avg_num_Proc/s', 'Std_num_proc/s'
           , 'Std_user_time', 'Avg_kbmemused', 'byte_rate', 'Std_rtps', 'Std_tps', 'PktRate', 'Des_ip_bytes', 'Avg_wtps', 'SrcBytes', 'Avg_tps' ,'SrcPort', 'Duration', 'TotalBytes', 'DstBytes', 'Avg_rtps', 'Std_ideal_time'
        # ,'Avg_ldavg_1', 'Std_system_time', 'Scr_packts_ratio', 'Des_pkts_ratio', 'File_activity', 'anomaly_alert', 'is_privileged', 'Process_activity', 'Succesful_login', 'Login_attempt', 'OSSEC_alert_level', 'OSSEC_alert', 'DstPort', 'read_write_physical.process', 'Avg_nice_time', 'Scr_bytes_ratio', 'Des_bytes_ratio'

]
df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)

label_col    = 'class2'
cat_cols     = ['Protocol','Service','Conn_state']
feature_cols = [c for c in df.columns if c not in [label_col] + cat_cols]

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
X_num = SimpleImputer(strategy='median').fit_transform(df[feature_cols])

df[cat_cols] = df[cat_cols].fillna('missing')
X_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore') \
    .fit_transform(df[cat_cols])

X = np.hstack([X_num, X_cat])
le = LabelEncoder()
y = le.fit_transform(df[label_col].fillna('missing'))
print("class2 映射：", dict(zip(le.classes_, le.transform(le.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test   = scaler.transform(X_test)
seq_len   = X_train.shape[1]
feat_dim  = 1
X_train_seq = X_train.reshape(-1, seq_len, feat_dim)
X_test_seq  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# Define Teacher Model
def build_teacher():
    inp = layers.Input(shape=(seq_len, feat_dim), name='teacher_input')
    # Two standard 1D convolutions
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv1')(inp)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False),
        name='bilstm'
    )(x)
    # Classification head
    out = layers.Dense(
        num_classes,
        activation='softmax',
        name='teacher_output'
    )(x)

    model = models.Model(inputs=inp, outputs=out, name='Teacher_BiLSTM')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

teacher = build_teacher()
mem_t0 = proc.memory_info().rss / (1024**2)
t0 = time.time()
teacher.fit(
    X_train_seq, y_train,
    validation_split=0.1,
    epochs=1, batch_size=256, verbose=1
)
print(f"Teacher train time: {time.time()-t0:.2f}s, RAM Δ: {proc.memory_info().rss/1024**2 - mem_t0:.2f} MB")
te_loss, te_acc = teacher.evaluate(X_test_seq, y_test, verbose=0)
print(f"Teacher eval loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# Soft Label
T = 10.0
train_logits = teacher.predict(X_train_seq, batch_size=512)
soft_train   = tf.nn.softmax(train_logits / T, axis=1)
test_logits  = teacher.predict(X_test_seq, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T, axis=1)

# Distillation Pipeline
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_seq, y_train, soft_train)
).map(lambda x, y, s: (x, (y, s))) \
 .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_test_seq, y_test, soft_test)
).map(lambda x, y, s: (x, (y, s))) \
 .batch(256).prefetch(tf.data.AUTOTUNE)

# Distiller 
class Distiller(models.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.student_loss_tracker      = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.accuracy_tracker          = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.accuracy_tracker
        ]

    def compile(self,
                optimizer,
                student_loss_fn,
                distillation_loss_fn,
                alpha=0.1,
                temperature=10):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distillation_loss_fn
        self.alpha           = alpha
        self.temperature     = temperature

    def train_step(self, data):
        x, (y_true, y_soft) = data
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            teacher_pred = self.teacher(x, training=False)
            loss_hard = self.student_loss_fn(y_true, student_pred)
            loss_soft = self.distill_loss_fn(
                tf.nn.softmax(teacher_pred/self.temperature, axis=1),
                tf.nn.softmax(student_pred/self.temperature, axis=1)
            )
            loss = self.alpha * loss_hard + (1-self.alpha) * loss_soft
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.student_loss_tracker.update_state(loss_hard)
        self.distillation_loss_tracker.update_state(loss_soft)
        self.accuracy_tracker.update_state(y_true, student_pred)
        return {
            "student_loss":      self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "accuracy":          self.accuracy_tracker.result(),
        }

    def test_step(self, data):
        x, (y_true, y_soft) = data
        student_pred = self.student(x, training=False)
        loss_hard = self.student_loss_fn(y_true, student_pred)
        self.student_loss_tracker.update_state(loss_hard)
        self.accuracy_tracker.update_state(y_true, student_pred)
        return {
            "student_loss": self.student_loss_tracker.result(),
            "accuracy":     self.accuracy_tracker.result(),
        }

# Student Model
def build_student():
    inp = layers.Input(shape=(seq_len, feat_dim), name='student_input')
    # Depthwise separable convolutions
    x = layers.SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='sep_conv1'
    )(inp)
    x = layers.SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='sep_conv2'
    )(x)
    # Compact bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False),
        name='student_bilstm'
    )(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    # Classification head
    out = layers.Dense(
        num_classes,
        activation='softmax',
        name='student_output'
    )(x)

    model = models.Model(inputs=inp, outputs=out, name='Student_CompactBiLSTM')
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

# Build student and measure its loaded memory
student = build_student()
mem_model = proc.memory_info().rss / (1024**2)
print(f"Student model loaded RAM: {mem_model:.2f} MB")

distiller = Distiller(student, teacher)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=T
)

mem_s0 = proc.memory_info().rss / (1024**2)
t1     = time.time()
distiller.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
print(f"Student distill time: {time.time()-t1:.2f}s, RAM Δ: {proc.memory_info().rss/1024**2 - mem_s0:.2f} MB")

# Evaluation
mem_inf0 = proc.memory_info().rss / (1024**2)

start_inf = time.time()
y_prob_inf = student.predict(X_test_seq, batch_size=256, verbose=0)
inf_time = time.time() - start_inf

mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test_seq.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")

y_pred = np.argmax(y_prob_inf, axis=1)
print("\nClassification Report (Student):")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_inf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{le.classes_[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.tight_layout(); plt.show()
