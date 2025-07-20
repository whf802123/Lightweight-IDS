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

# ==============================
# 监控进程
# ==============================
proc = psutil.Process(os.getpid())

# ==============================
# 1. 读入 CSV，把 '?' 和 '-' 当成 NaN
# ==============================
df = pd.read_csv(
    r'C:\Users\Administrator\Desktop\X-IIoTID dataset.csv',
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==============================
# 2. 布尔列 'TRUE'/'FALSE' → 1/0
# ==============================
bool_cols = [
    'is_syn_only','Is_SYN_ACK','is_pure_ack','is_with_payload',
    'FIN or RST','Bad_checksum','is_SYN_with_RST','anomaly_alert'
]
for c in bool_cols:
    if c in df.columns:
        df[c] = df[c].map({'TRUE':1, 'FALSE':0})

# ==============================
# 3. 丢弃不需要的列
# ==============================
drop_cols = ['Date','Timestamp','SrcIP','DstIP','class1','class3'

            , 'Std_nice_time', 'DstPkts', 'Scr_ip_bytes', 'TotalPkts, Std_ldavg_1', 'Std_kbmemused', 'SrcPkts', 'Std_wtps', 'Avg_iowait_time', 'Std_iowait_time', 'missed_bytes', 'Avg_num_Proc/s', 'Std_num_proc/s'
            , 'Std_user_time', 'Avg_kbmemused', 'byte_rate', 'Std_rtps', 'Std_tps', 'PktRate', 'Des_ip_bytes', 'Avg_wtps', 'SrcBytes', 'Avg_tps' ,'SrcPort', 'Duration', 'TotalBytes', 'DstBytes', 'Avg_rtps', 'Std_ideal_time'

        #  ,'Avg_ldavg_1', 'Std_system_time', 'Scr_packts_ratio', 'Des_pkts_ratio', 'File_activity', 'anomaly_alert', 'is_privileged', 'Process_activity', 'Succesful_login', 'Login_attempt', 'OSSEC_alert_level', 'OSSEC_alert', 'DstPort', 'read_write_physical.process', 'Avg_nice_time', 'Scr_bytes_ratio', 'Des_bytes_ratio'

]
df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)

# ==============================
# 4. 准备特征和标签
# ==============================
label_col    = 'class2'
cat_cols     = ['Protocol','Service','Conn_state']
feature_cols = [c for c in df.columns if c not in [label_col] + cat_cols]

# 数值型 → float → 中位数插补
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
X_num = SimpleImputer(strategy='median').fit_transform(df[feature_cols])

# 类别型 → 填 'missing' + One-Hot
df[cat_cols] = df[cat_cols].fillna('missing')
X_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore') \
    .fit_transform(df[cat_cols])

X = np.hstack([X_num, X_cat])
le = LabelEncoder()
y = le.fit_transform(df[label_col].fillna('missing'))
print("class2 映射：", dict(zip(le.classes_, le.transform(le.classes_))))

# ==============================
# 5. 划分训练/测试 & 标准化
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test   = scaler.transform(X_test)

# —— 把特征当作时间序列长度 seq_len，feat_dim=1 ——
seq_len   = X_train.shape[1]
feat_dim  = 1
X_train_seq = X_train.reshape(-1, seq_len, feat_dim)
X_test_seq  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# ==============================
# 6. 定义教师模型
# ==============================
def build_teacher():
    inp = layers.Input((seq_len, feat_dim))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)

    def tcn_block(x, rate):
        conv = layers.Conv1D(64, 3, padding='causal',
                             dilation_rate=rate, activation='relu')(x)
        conv = layers.BatchNormalization()(conv)
        return layers.Add()([x, conv])

    for r in [1, 2]:
        x = tcn_block(x, r)

    scores  = layers.Dense(1, activation='tanh')(x)
    weights = layers.Softmax(axis=1)(scores)
    weighted = layers.Multiply()([x, weights])
    ctx      = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(weighted)
    out      = layers.Dense(num_classes, activation='softmax')(ctx)

    model = models.Model(inp, out, name='Teacher')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
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

# ==============================
# 7. 生成软标签
# ==============================
T = 10.0
train_logits = teacher.predict(X_train_seq, batch_size=512)
soft_train   = tf.nn.softmax(train_logits / T, axis=1)
test_logits  = teacher.predict(X_test_seq, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T, axis=1)

# ==============================
# 8. 构建蒸馏数据管道
# ==============================
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_seq, y_train, soft_train)
).map(lambda x, y, s: (x, (y, s))) \
 .cache().shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_test_seq, y_test, soft_test)
).map(lambda x, y, s: (x, (y, s))) \
 .batch(256).prefetch(tf.data.AUTOTUNE)

# ==============================
# 9. 定义 Distiller
# ==============================
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

# ==============================
# 10. 构建学生模型 & 蒸馏训练
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
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model

# Build student and measure its loaded memory
student = build_student()


mem_before = proc.memory_info().rss / (1024**2)
t0 = time.time()

# 3. 纯学生模型训练
history = student.fit(
    X_train_seq, y_train,
    validation_data=(X_test_seq, y_test),  # 或者用 validation_split=0.1
    epochs=1,
    batch_size=256,
    verbose=1
)

# 4. 计算耗时与内存增量
train_time = time.time() - t0
mem_after = proc.memory_info().rss / (1024**2)
print(f"Student-only train time: {train_time:.2f}s, RAM Δ: {mem_after - mem_before:.2f} MB")



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

# 蒸馏训练内存增量
mem_s0 = proc.memory_info().rss / (1024**2)
t1     = time.time()
distiller.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
print(f"Student distill time: {time.time()-t1:.2f}s, RAM Δ: {proc.memory_info().rss/1024**2 - mem_s0:.2f} MB")

# ==============================
# 11. 最终评估学生模型 + 计算推理时间和内存
# ==============================

# 1) 记录推理前内存
mem_inf0 = proc.memory_info().rss / (1024**2)

# 2) 计时并执行推理
start_inf = time.time()
y_prob_inf = student.predict(X_test_seq, batch_size=256, verbose=0)
inf_time = time.time() - start_inf

# 3) 记录推理后内存
mem_inf1 = proc.memory_info().rss / (1024**2)

# 4) 计算样本数并打印结果
n_samples = X_test_seq.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")

# 5) 后续常规评估
y_pred = np.argmax(y_prob_inf, axis=1)
print("\nClassification Report (Student):")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ROC 曲线及 AUC
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



