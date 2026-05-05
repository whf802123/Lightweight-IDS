
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import psutil
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, losses, optimizers, metrics as keras_metrics
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Pruning
from tensorflow_model_optimization.sparsity import keras as sparsity

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)   # XLA JIT -> operator fusion
tf.random.set_seed(42)
np.random.seed(42)

# Optional: GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth setup failed:", e)

cpus = tf.config.list_physical_devices('CPU')
print("Available CPU devices:", cpus)
print("Available GPU devices:", gpus)

proc = psutil.Process(os.getpid())

df = pd.read_csv(
    r'C:\Users\whf80\Desktop\Lightweight\Dataset\wustl_iiot\wustl_iiot_1.csv',
    dtype=str,
    na_values=['?', '-'],
    keep_default_na=True,
    low_memory=False
)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

drop_cols = [
    'Timestamp', 'LastTime', 'SrcIP', 'DstIP', 'Target', 'sIpId', 'dIpId'
]
df.drop(columns=drop_cols, errors='ignore', inplace=True)

X = df.drop(columns=['Traffic']).apply(pd.to_numeric, errors='coerce')
y = df['Traffic'].fillna('missing').astype(str)

X_imp = SimpleImputer(strategy='median').fit_transform(X)

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_enc, test_size=0.3, stratify=y_enc, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

seq_len = X_train.shape[1]
feat_dim = 1
num_classes = len(le.classes_)
teacher_att_len = seq_len // 2  # after MaxPooling1D(2) in teacher

X_train = X_train.reshape(-1, seq_len, feat_dim).astype(np.float32)
X_test = X_test.reshape(-1, seq_len, feat_dim).astype(np.float32)

print(f"seq_len={seq_len}, num_classes={num_classes}, teacher_att_len={teacher_att_len}")

teacher_es = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1
)

distill_es = EarlyStopping(
    monitor='val_student_loss',
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1
)

@tf.function(jit_compile=True)
def compute_attention_map(x):
    """
    x: [batch, length, channels]
    return: normalized attention map [batch, length, 1]
    """
    att = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    att = att / (tf.reduce_sum(att, axis=1, keepdims=True) + 1e-8)
    return att

def build_teacher_and_attention():
    inp = layers.Input(shape=(seq_len, feat_dim), name="teacher_input")

    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='t_conv1')(inp)
    x = layers.MaxPooling1D(2, name='t_pool')(x)

    def tcn_block(x_in, rate, name_prefix):
        conv = layers.Conv1D(
            64, 3, padding='causal', dilation_rate=rate,
            activation='relu', name=f'{name_prefix}_conv'
        )(x_in)
        conv = layers.BatchNormalization(name=f'{name_prefix}_bn')(conv)
        out = layers.Add(name=f'{name_prefix}_add')([x_in, conv])
        return out

    for rate in [1, 2]:
        x = tcn_block(x, rate, f"t_tcn_r{rate}")

    feat_map = layers.Conv1D(32, 1, activation='relu', name='t_feat_map')(x)

    # classification branch
    scores = layers.Dense(1, activation='tanh', name='t_scores')(feat_map)
    weights = layers.Softmax(axis=1, name='t_weights')(scores)
    weighted = layers.Multiply(name='t_weighted')([feat_map, weights])
    ctx = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1), name='t_ctx')(weighted)
    out = layers.Dense(num_classes, activation='softmax', name='t_output')(ctx)

    # attention branch
    att_map = layers.Lambda(compute_attention_map, name='TeacherAttention')(feat_map)

    teacher_model = models.Model(inp, out, name='Teacher')
    teacher_att_model = models.Model(inp, att_map, name='TeacherAttentionModel')

    teacher_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.SparseCategoricalAccuracy(name='accuracy')],
        jit_compile=True
    )

    return teacher_model, teacher_att_model

teacher, teacher_att = build_teacher_and_attention()

mem_t0 = proc.memory_info().rss / (1024 ** 2)
t0 = time.time()

teacher.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=256,
    callbacks=[teacher_es],
    verbose=1
)

teacher_train_time = time.time() - t0
mem_t1 = proc.memory_info().rss / (1024 ** 2)

print(f"Teacher training time: {teacher_train_time:.2f}s")
print(f"Teacher RAM Δ: {mem_t1 - mem_t0:.2f} MB")

te_loss, te_acc = teacher.evaluate(X_test, y_test, verbose=0)
print(f"Teacher eval loss: {te_loss:.4f}, acc: {te_acc:.4f}")

T = 10.0

teacher_prob_train = teacher.predict(X_train, batch_size=512, verbose=0)
teacher_prob_test = teacher.predict(X_test, batch_size=512, verbose=0)

# Use softened probabilities for KD
soft_train = tf.nn.softmax(
    tf.math.log(tf.clip_by_value(teacher_prob_train, 1e-7, 1.0)) / T,
    axis=1
)
soft_test = tf.nn.softmax(
    tf.math.log(tf.clip_by_value(teacher_prob_test, 1e-7, 1.0)) / T,
    axis=1
)

soft_train = tf.cast(soft_train, tf.float32)
soft_test = tf.cast(soft_test, tf.float32)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, soft_train))
train_ds = (
    train_ds
    .shuffle(10000)
    .batch(256)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test, soft_test))
val_ds = (
    val_ds
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

# Student model
def build_base_student():
    inp = layers.Input(shape=(seq_len, feat_dim), name="student_input")

    x = layers.GRU(64, implementation=2, name='s_gru')(inp)
    x = layers.Dropout(0.3, name='s_dropout')(x)

    # classification branch
    out = layers.Dense(num_classes, activation='softmax', name='s_output')(x)

    # attention branch:
    # project GRU output into a sequence-like feature map
    att_feat = layers.Dense(teacher_att_len * 32, activation='relu', name='s_att_proj')(x)
    att_feat = layers.Reshape((teacher_att_len, 32), name='s_feat_map')(att_feat)
    att_map = layers.Lambda(compute_attention_map, name='StudentAttention')(att_feat)

    model = models.Model(inp, [out, att_map], name='Student_AT')
    return model

base_student = build_base_student()

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

student = sparsity.prune_low_magnitude(base_student, **pruning_params)

# Compile only for graph building / predict friendliness
student.compile(
    optimizer='adam',
    loss=None,
    jit_compile=True
)

print("\nPrunable student model built.")


#  Attention Transfer 
class DistillerAT(models.Model):
    def __init__(self, student, teacher, teacher_att, alpha=0.5, beta=0.1, temp=10.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_att = teacher_att
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

        self.student_loss_tracker = keras_metrics.Mean(name="student_loss")
        self.distill_loss_tracker = keras_metrics.Mean(name="distillation_loss")
        self.att_loss_tracker = keras_metrics.Mean(name="attention_loss")
        self.total_loss_tracker = keras_metrics.Mean(name="total_loss")
        self.accuracy_tracker = keras_metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.student_loss_tracker,
            self.distill_loss_tracker,
            self.att_loss_tracker,
            self.total_loss_tracker,
            self.accuracy_tracker
        ]

    def compile(self, optimizer, student_loss_fn, distill_loss_fn, att_loss_fn):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.att_loss_fn = att_loss_fn

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y_true, y_soft = data

        with tf.GradientTape() as tape:
            student_pred, att_s = self.student(x, training=True)
            teacher_pred = self.teacher(x, training=False)
            att_t = self.teacher_att(x, training=False)

            # hard loss
            loss_hard = self.student_loss_fn(y_true, student_pred)

            # soft KD loss
            student_soft = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(student_pred, 1e-7, 1.0)) / self.temp,
                axis=1
            )
            loss_soft = self.distill_loss_fn(y_soft, student_soft)

            # AT loss
            loss_att = self.att_loss_fn(att_t, att_s)

            total_loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft + self.beta * loss_att

        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.student_loss_tracker.update_state(loss_hard)
        self.distill_loss_tracker.update_state(loss_soft)
        self.att_loss_tracker.update_state(loss_att)
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y_true, student_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        x, y_true, y_soft = data

        student_pred, att_s = self.student(x, training=False)
        att_t = self.teacher_att(x, training=False)

        loss_hard = self.student_loss_fn(y_true, student_pred)

        student_soft = tf.nn.softmax(
            tf.math.log(tf.clip_by_value(student_pred, 1e-7, 1.0)) / self.temp,
            axis=1
        )
        loss_soft = self.distill_loss_fn(y_soft, student_soft)
        loss_att = self.att_loss_fn(att_t, att_s)

        total_loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft + self.beta * loss_att

        self.student_loss_tracker.update_state(loss_hard)
        self.distill_loss_tracker.update_state(loss_soft)
        self.att_loss_tracker.update_state(loss_att)
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y_true, student_pred)

        return {m.name: m.result() for m in self.metrics}

distiller = DistillerAT(
    student=student,
    teacher=teacher,
    teacher_att=teacher_att,
    alpha=0.5,
    beta=0.1,
    temp=T
)

distiller.compile(
    optimizer=optimizers.Adam(),
    student_loss_fn=losses.SparseCategoricalCrossentropy(),
    distill_loss_fn=losses.KLDivergence(),
    att_loss_fn=losses.MeanSquaredError()
)

# pruning callback is required
prune_callbacks = [
    sparsity.UpdatePruningStep(),
    distill_es
]


# Distillation 
print("\n=== Distillation + AT + Pruning Training ===")
mem_s0 = proc.memory_info().rss / (1024 ** 2)
d0 = time.time()

distiller.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=prune_callbacks,
    verbose=1
)

distill_train_time = time.time() - d0
mem_s1 = proc.memory_info().rss / (1024 ** 2)

print(f"Distillation training time: {distill_train_time:.2f}s")
print(f"Distillation RAM Δ: {mem_s1 - mem_s0:.2f} MB")

student_stripped = sparsity.strip_pruning(student)
print("\nPruning wrappers stripped for final inference model.")


# Evaluation
print("\n=== Final Evaluation ===")
wall_before = time.time()
cpu_before = proc.cpu_times().user + proc.cpu_times().system
ram_before = proc.memory_info().rss

y_prob, y_att = student_stripped.predict(X_test, batch_size=512, verbose=0)
y_labels = np.argmax(y_prob, axis=1)

wall_after = time.time()
cpu_after = proc.cpu_times().user + proc.cpu_times().system
ram_after = proc.memory_info().rss

print(f"Wall time: {wall_after - wall_before:.2f}s")
print(f"CPU time:  {cpu_after - cpu_before:.2f}s")
print(f"RAM Δ:     {(ram_after - ram_before) / 1024**2:.2f} MB")

print("\nClassification Report:")
print(classification_report(y_test, y_labels, target_names=le.classes_, digits=4))


# Confusion Matrix
cm = confusion_matrix(y_test, y_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.tight_layout()
plt.show()


#  ROC Curve
y_test_bin = label_binarize(y_test, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
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
