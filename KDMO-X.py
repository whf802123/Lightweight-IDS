import os
import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, metrics, losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    label_binarize
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)   

tf.random.set_seed(42)
np.random.seed(42)

proc = psutil.Process(os.getpid())

df = pd.read_csv(
    r'C:\Users\whf80\Desktop\Lightweight\Dataset\x-IIoTID\X-IIoTID dataset.csv',
    dtype=str,
    na_values=['?', '-'],
    keep_default_na=True,
    low_memory=False
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

bool_cols = [
    'is_syn_only', 'Is_SYN_ACK', 'is_pure_ack', 'is_with_payload',
    'FIN or RST', 'Bad_checksum', 'is_SYN_with_RST', 'anomaly_alert'
]
for c in bool_cols:
    if c in df.columns:
        df[c] = df[c].map({'TRUE': 1, 'FALSE': 0})

drop_cols = ['Date', 'Timestamp', 'SrcIP', 'DstIP', 'class1', 'class3']
df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)


label_col = 'class2'
cat_cols = ['Protocol', 'Service', 'Conn_state']
feature_cols = [c for c in df.columns if c not in [label_col] + cat_cols]

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

X_num = SimpleImputer(strategy='median').fit_transform(df[feature_cols])

df[cat_cols] = df[cat_cols].fillna('missing')
X_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df[cat_cols])

X = np.hstack([X_num, X_cat])

le = LabelEncoder()
y = le.fit_transform(df[label_col].fillna('missing'))
print("class2 ：", dict(zip(le.classes_, le.transform(le.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

seq_len = X_train.shape[1]
feat_dim = 1
X_train_seq = X_train.reshape(-1, seq_len, feat_dim).astype(np.float32)
X_test_seq = X_test.reshape(-1, seq_len, feat_dim).astype(np.float32)
num_classes = len(le.classes_)


# Early stopping

es_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1
)

# Attention map

@tf.function(jit_compile=True)
def compute_attention_map(x):
    att = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    att = att / (tf.reduce_sum(att, axis=1, keepdims=True) + 1e-8)
    return att


def build_teacher_with_attention():
    inp = layers.Input((seq_len, feat_dim), name="teacher_input")

    x = layers.Conv1D(64, 3, padding='same', activation='relu', name="t_conv1")(inp)
    x = layers.MaxPooling1D(2, name="t_pool1")(x)

    def tcn_block(x_in, rate, name_prefix):
        conv = layers.Conv1D(
            64,
            3,
            padding='causal',
            dilation_rate=rate,
            activation='relu',
            name=f"{name_prefix}_conv"
        )(x_in)
        conv = layers.BatchNormalization(name=f"{name_prefix}_bn")(conv)
        out = layers.Add(name=f"{name_prefix}_add")([x_in, conv])
        return out

    for r in [1, 2]:
        x = tcn_block(x, r, f"t_tcn_r{r}")

    feat_map = layers.Conv1D(32, 1, activation='relu', name='t_feat_map')(x)
    ctx = layers.GlobalAveragePooling1D(name='t_gap')(feat_map)
    out = layers.Dense(num_classes, activation='softmax', name='t_output')(ctx)

    att_map = layers.Lambda(compute_attention_map, name='TeacherAttention')(feat_map)

    model = models.Model(inp, out, name='Teacher')
    att_model = models.Model(inp, att_map, name='TeacherAttentionModel')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model, att_model


# 9. Student model 


def build_student_with_attention():
    inp = layers.Input((seq_len, feat_dim), name="student_input")

    x = layers.GRU(64, implementation=2, name='s_gru')(inp)
    feat_map = layers.Reshape((1, 64), name='s_feat_map')(x)
    out = layers.Dense(num_classes, activation='softmax', name='s_output')(x)

    att_map = layers.Lambda(compute_attention_map, name='StudentAttention')(feat_map)

    model = models.Model(inp, out, name='Student')
    att_model = models.Model(inp, att_map, name='StudentAttentionModel')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )
    return model, att_model


# Distiller with Attention Transfer

class DistillerAT(models.Model):
    def __init__(self, student, teacher, student_att, teacher_att,
                 alpha=0.1, beta=0.1, temperature=10.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.student_att = student_att
        self.teacher_att = teacher_att
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.student_loss_tracker = metrics.Mean(name="student_loss")
        self.distill_loss_tracker = metrics.Mean(name="distillation_loss")
        self.att_loss_tracker = metrics.Mean(name="attention_loss")
        self.accuracy_tracker = metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.student_loss_tracker,
            self.distill_loss_tracker,
            self.att_loss_tracker,
            self.accuracy_tracker
        ]

    def compile(self, optimizer, student_loss_fn, distill_loss_fn, att_loss_fn):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.att_loss_fn = att_loss_fn

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, (y_true, y_soft) = data

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            teacher_pred = self.teacher(x, training=False)

            loss_hard = self.student_loss_fn(y_true, student_pred)

            teacher_soft = tf.nn.softmax(teacher_pred / self.temperature, axis=1)
            student_soft = tf.nn.softmax(student_pred / self.temperature, axis=1)
            loss_soft = self.distill_loss_fn(teacher_soft, student_soft)

            att_s = self.student_att(x, training=True)
            att_t = self.teacher_att(x, training=False)

            # Resize student attention map to teacher attention length if needed
            if att_s.shape[1] != att_t.shape[1]:
                att_s = tf.image.resize(att_s, size=(att_t.shape[1], 1), method='bilinear')

            loss_att = self.att_loss_fn(att_t, att_s)

            loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft + self.beta * loss_att

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.student_loss_tracker.update_state(loss_hard)
        self.distill_loss_tracker.update_state(loss_soft)
        self.att_loss_tracker.update_state(loss_att)
        self.accuracy_tracker.update_state(y_true, student_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        x, (y_true, y_soft) = data

        student_pred = self.student(x, training=False)
        loss_hard = self.student_loss_fn(y_true, student_pred)

        teacher_pred = self.teacher(x, training=False)
        teacher_soft = tf.nn.softmax(teacher_pred / self.temperature, axis=1)
        student_soft = tf.nn.softmax(student_pred / self.temperature, axis=1)
        loss_soft = self.distill_loss_fn(teacher_soft, student_soft)

        att_s = self.student_att(x, training=False)
        att_t = self.teacher_att(x, training=False)

        if att_s.shape[1] != att_t.shape[1]:
            att_s = tf.image.resize(att_s, size=(att_t.shape[1], 1), method='bilinear')

        loss_att = self.att_loss_fn(att_t, att_s)

        self.student_loss_tracker.update_state(loss_hard)
        self.distill_loss_tracker.update_state(loss_soft)
        self.att_loss_tracker.update_state(loss_att)
        self.accuracy_tracker.update_state(y_true, student_pred)

        return {m.name: m.result() for m in self.metrics}

teacher, teacher_att = build_teacher_with_attention()

mem_t0 = proc.memory_info().rss / (1024 ** 2)
t0 = time.time()

teacher.fit(
    X_train_seq,
    y_train,
    validation_split=0.1,
    epochs=50,              # EarlyStopping can now really work
    batch_size=256,
    callbacks=[es_callback],
    verbose=1
)

teacher_time = time.time() - t0
mem_t1 = proc.memory_info().rss / (1024 ** 2)

print(f"Teacher training time: {teacher_time:.2f} s")
print(f"Teacher RAM Δ: {mem_t1 - mem_t0:.2f} MB")

teacher_eval_loss, teacher_eval_acc = teacher.evaluate(X_test_seq, y_test, verbose=0)
print(f"Teacher eval loss: {teacher_eval_loss:.4f}, acc: {teacher_eval_acc:.4f}")


T = 10.0

train_logits = teacher.predict(X_train_seq, batch_size=512, verbose=0)
soft_train = tf.nn.softmax(train_logits / T, axis=1)

test_logits = teacher.predict(X_test_seq, batch_size=512, verbose=0)
soft_test = tf.nn.softmax(test_logits / T, axis=1)

train_ds = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train, soft_train))
train_ds = (
    train_ds
    .map(lambda x, y, s: (x, (y, s)), num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_test_seq, y_test, soft_test))
val_ds = (
    val_ds
    .map(lambda x, y, s: (x, (y, s)), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

student, student_att = build_student_with_attention()

mem_model = proc.memory_info().rss / (1024 ** 2)
print(f"Student model loaded RAM: {mem_model:.2f} MB")

distiller = DistillerAT(
    student=student,
    teacher=teacher,
    student_att=student_att,
    teacher_att=teacher_att,
    alpha=0.1,
    beta=0.1,
    temperature=T
)

distiller.compile(
    optimizer=optimizers.Adam(),
    student_loss_fn=losses.SparseCategoricalCrossentropy(),
    distill_loss_fn=losses.KLDivergence(),
    att_loss_fn=losses.MeanSquaredError()
)

mem_s0 = proc.memory_info().rss / (1024 ** 2)
t1 = time.time()

distiller.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,        
    callbacks=[es_callback],
    verbose=1
)

student_time = time.time() - t1
mem_s1 = proc.memory_info().rss / (1024 ** 2)

print(f"Distillation training time: {student_time:.2f} s")
print(f"Distillation RAM Δ: {mem_s1 - mem_s0:.2f} MB")

# Evaluation

mem_inf0 = proc.memory_info().rss / (1024 ** 2)

start_inf = time.time()
y_prob_inf = student.predict(X_test_seq, batch_size=256, verbose=0)
inf_time = time.time() - start_inf

mem_inf1 = proc.memory_info().rss / (1024 ** 2)

n_samples = X_test_seq.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time / n_samples * 1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")

y_pred = np.argmax(y_prob_inf, axis=1)

print("\nClassification Report (Student):")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ROC 

y_test_bin = label_binarize(y_test, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_inf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{le.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
