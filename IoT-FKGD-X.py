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
from tensorflow.keras import layers, models, metrics, losses, optimizers

# ==============================
# Process monitor
# ==============================
proc = psutil.Process(os.getpid())

# ==============================
# 1. Load CSV and treat '?' and '-' as NaN
# ==============================
df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\X-IIoTID dataset.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==============================
# 2. Convert boolean columns 'TRUE'/'FALSE' to 1/0
# ==============================
bool_cols = [
    'is_syn_only','Is_SYN_ACK','is_pure_ack','is_with_payload',
    'FIN or RST','Bad_checksum','is_SYN_with_RST','anomaly_alert'
]
for c in bool_cols:
    if c in df.columns:
        df[c] = df[c].map({'TRUE':1, 'FALSE':0})

# ==============================
# 3. Drop unused columns
# ==============================
drop_cols = ['Date','Timestamp','SrcIP','DstIP','class1','class3'

          # , 'Std_nice_time', 'DstPkts', 'Scr_ip_bytes', 'TotalPkts, Std_ldavg_1', 'Std_kbmemused', 'SrcPkts', 'Std_wtps', 'Avg_iowait_time', 'Std_iowait_time', 'missed_bytes', 'Avg_num_Proc/s', 'Std_num_proc/s'
           # , 'Std_user_time', 'Avg_kbmemused', 'byte_rate', 'Std_rtps', 'Std_tps', 'PktRate', 'Des_ip_bytes', 'Avg_wtps', 'SrcBytes', 'Avg_tps' ,'SrcPort', 'Duration', 'TotalBytes', 'DstBytes', 'Avg_rtps', 'Std_ideal_time'
          #  ,'Avg_ldavg_1', 'Std_system_time', 'Scr_packts_ratio', 'Des_pkts_ratio', 'File_activity', 'anomaly_alert', 'is_privileged', 'Process_activity', 'Succesful_login', 'Login_attempt', 'OSSEC_alert_level', 'OSSEC_alert', 'DstPort', 'read_write_physical.process', 'Avg_nice_time', 'Scr_bytes_ratio', 'Des_bytes_ratio'

]
df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)

# ==============================
# 4. Prepare features and labels
# ==============================
label_col    = 'class2'
cat_cols     = ['Protocol','Service','Conn_state']
feature_cols = [c for c in df.columns if c not in [label_col] + cat_cols]

# Numeric features → float → median imputation
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
X_num = SimpleImputer(strategy='median').fit_transform(df[feature_cols])

# Categorical features → fill 'missing' + one-hot encoding
df[cat_cols] = df[cat_cols].fillna('missing')
X_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore') \
    .fit_transform(df[cat_cols])

X = np.hstack([X_num, X_cat])
le = LabelEncoder()
y = le.fit_transform(df[label_col].fillna('missing'))
print("class2 mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ==============================
# 5. Train / test split & standardization
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test   = scaler.transform(X_test)

# —— Treat features as time steps: seq_len, feat_dim = 1 ——
seq_len     = X_train.shape[1]
feat_dim    = 1
X_train_seq = X_train.reshape(-1, seq_len, feat_dim)
X_test_seq  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# ==============================
# 6. Define IoT-FKGD teacher & student architectures
# ==============================
class LSHSelfAttention(layers.Layer):
    def __init__(self, num_hashes, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_hashes = num_hashes
        self.key_dim = key_dim
    def build(self, input_shape):
        self.q = layers.Dense(self.key_dim, use_bias=False)
        self.k = layers.Dense(self.key_dim, use_bias=False)
        self.v = layers.Dense(self.key_dim, use_bias=False)
        super().build(input_shape)
    def call(self, x):
        qx = self.q(x)
        kx = self.k(x)
        vx = self.v(x)
        # TODO: implement true LSH bucketing on qx/kx
        return layers.Attention()([qx, kx, vx])

class GraphConv(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight((input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)
    def call(self, x, adj=None):
        return tf.matmul(x, self.w)

def build_teacher_iotfkgd():
    inp = layers.Input((seq_len, feat_dim))
    # Multi-scale dilated convolutions
    c1 = layers.Conv1D(128, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    c2 = layers.Conv1D(128, 3, padding='causal', dilation_rate=2, activation='relu')(inp)
    c4 = layers.Conv1D(128, 3, padding='causal', dilation_rate=4, activation='relu')(inp)
    x  = layers.Concatenate()([c1, c2, c4])
    # LSH self-attention
    x  = LSHSelfAttention(num_hashes=8, key_dim=256)(x)
    # GCN hidden 2048
    x  = GraphConv(2048)(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.ReLU()(x)
    # Pooling + MLP
    x  = layers.GlobalAveragePooling1D()(x)
    x  = layers.Dense(2048, activation='relu')(x)
    out= layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='Teacher_IoT_FKGD')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[metrics.SparseCategoricalAccuracy()])
    return model

def build_student_iotfkgd():
    inp = layers.Input((seq_len, feat_dim))
    c1 = layers.Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    c2 = layers.Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(inp)
    c4 = layers.Conv1D(64, 3, padding='causal', dilation_rate=4, activation='relu')(inp)
    x  = layers.Concatenate()([c1, c2, c4])
    x  = LSHSelfAttention(num_hashes=4, key_dim=128)(x)
    x  = GraphConv(512)(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.ReLU()(x)
    x  = layers.GlobalAveragePooling1D()(x)
    x  = layers.Dense(512, activation='relu')(x)
    out= layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='Student_IoT_FKGD')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  jit_compile=True)
    return model

teacher = build_teacher_iotfkgd()
student = build_student_iotfkgd()

# ==============================
# 7. Train Teacher
# ==============================
mem_t0 = proc.memory_info().rss/1024**2
t0 = time.time()
teacher.fit(X_train_seq, y_train, validation_split=0.1, epochs=1, batch_size=256, verbose=1)
print(f"Teacher train time: {time.time()-t0:.2f}s, RAM Δ: {proc.memory_info().rss/1024**2 - mem_t0:.2f} MB")
te_loss, te_acc = teacher.evaluate(X_test_seq, y_test, verbose=0)
print(f"Teacher eval loss: {te_loss:.4f}, acc: {te_acc:.4f}")

# ==============================
# 8. Generate soft labels & build data pipeline
# ==============================
T = 10.0
train_logits = teacher.predict(X_train_seq, batch_size=512)
soft_train   = tf.nn.softmax(train_logits/T, axis=1)
test_logits  = teacher.predict(X_test_seq,  batch_size=512)
soft_test    = tf.nn.softmax(test_logits/T, axis=1)

train_ds = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train, soft_train)) \
    .map(lambda x, y, s: (x, (y, s))) \
    .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test_seq, y_test, soft_test)) \
    .map(lambda x, y, s: (x, (y, s))) \
    .batch(256).prefetch(tf.data.AUTOTUNE)

# ==============================
# 9. Define Distiller
# ==============================
class Distiller(models.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.sl_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.dl_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.acc_tracker= tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    @property
    def metrics(self):
        return [self.sl_tracker, self.dl_tracker, self.acc_tracker]
    def compile(self, optimizer, student_loss_fn, distill_loss_fn, alpha=0.1, temperature=10):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    def train_step(self, data):
        x, (y_true, y_soft) = data
        with tf.GradientTape() as tape:
            s_pred = self.student(x, training=True)
            t_pred = self.teacher(x, training=False)
            loss_h = self.student_loss_fn(y_true, s_pred)
            loss_s = self.distill_loss_fn(
                tf.nn.softmax(t_pred/self.temperature, axis=1),
                tf.nn.softmax(s_pred/self.temperature, axis=1)
            )
            loss = self.alpha*loss_h + (1-self.alpha)*loss_s
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.sl_tracker.update_state(loss_h)
        self.dl_tracker.update_state(loss_s)
        self.acc_tracker.update_state(y_true, s_pred)
        return {"student_loss": self.sl_tracker.result(),
                "distillation_loss": self.dl_tracker.result(),
                "accuracy": self.acc_tracker.result()}
    def test_step(self, data):
        x, (y_true, _) = data
        s_pred = self.student(x, training=False)
        loss_h = self.student_loss_fn(y_true, s_pred)
        self.sl_tracker.update_state(loss_h)
        self.acc_tracker.update_state(y_true, s_pred)
        return {"student_loss": self.sl_tracker.result(),
                "accuracy": self.acc_tracker.result()}

# ==============================
# 10. Distillation training
# ==============================
distiller = Distiller(student, teacher)
distiller.compile(
    optimizer=optimizers.Adam(),
    student_loss_fn=losses.SparseCategoricalCrossentropy(),
    distill_loss_fn=losses.KLDivergence(),
    alpha=0.1,
    temperature=T
)
mem_s0 = proc.memory_info().rss/1024**2
t1 = time.time()
distiller.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
print(f"Distill train time: {time.time()-t1:.2f}s, RAM Δ: {proc.memory_info().rss/1024**2 - mem_s0:.2f} MB")

# ==============================
# 11. Final evaluation & inference profiling
# ==============================
mem_inf0 = proc.memory_info().rss/1024**2
start_inf = time.time()
y_prob = student.predict(X_test_seq, batch_size=256, verbose=0)
inf_time = time.time() - start_inf
mem_inf1 = proc.memory_info().rss/1024**2
n_samples = X_test_seq.shape[0]
print(f"Inference time: {inf_time:.4f}s for {n_samples} samples, avg {inf_time/n_samples*1e3:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")

y_pred = np.argmax(y_prob, axis=1)
print("\nClassification Report (Student):")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

y_test_bin = label_binarize(y_test, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{le.classes_[i]} (AUC={roc_auc[i]:.2f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.tight_layout()
plt.show()
