import os
import time
import psutil
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, metrics

# paths to your train/test CSVs
train_csv = r'C:\Users\Administrator\Desktop\X-IIoTID_10pct_train.csv'
test_csv  = r'C:\Users\Administrator\Desktop\X-IIoTID_10pct_test.csv'

# monitoring
proc = psutil.Process(os.getpid())
def get_mem_mb():
    return proc.memory_info().rss / 1024**2
random_state = 42

# LOAD DATA
df_train = pd.read_csv(
    train_csv,
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)
df_test = pd.read_csv(
    test_csv,
    dtype=str, na_values=['?','-'], keep_default_na=True, low_memory=False
)

# replace infinities
for df in (df_train, df_test):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

# boolean columns → 1/0
bool_cols = [
    'is_syn_only','Is_SYN_ACK','is_pure_ack','is_with_payload',
    'FIN or RST','Bad_checksum','is_SYN_with_RST','anomaly_alert'
]
for df in (df_train, df_test):
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map({'TRUE':1, 'FALSE':0})

# drop unwanted columns
drop_cols = ['Date','Timestamp','SrcIP','DstIP','class1','class3'

            , 'Std_nice_time', 'DstPkts', 'Scr_ip_bytes', 'TotalPkts, Std_ldavg_1', 'Std_kbmemused', 'SrcPkts', 'Std_wtps', 'Avg_iowait_time', 'Std_iowait_time', 'missed_bytes', 'Avg_num_Proc/s', 'Std_num_proc/s'
            , 'Std_user_time', 'Avg_kbmemused', 'byte_rate', 'Std_rtps', 'Std_tps', 'PktRate', 'Des_ip_bytes', 'Avg_wtps', 'SrcBytes', 'Avg_tps' ,'SrcPort', 'Duration', 'TotalBytes', 'DstBytes', 'Avg_rtps', 'Std_ideal_time'
          #  ,'Avg_ldavg_1', 'Std_system_time', 'Scr_packts_ratio', 'Des_pkts_ratio', 'File_activity', 'anomaly_alert', 'is_privileged', 'Process_activity', 'Succesful_login', 'Login_attempt', 'OSSEC_alert_level', 'OSSEC_alert', 'DstPort', 'read_write_physical.process', 'Avg_nice_time', 'Scr_bytes_ratio', 'Des_bytes_ratio'

]
for df in (df_train, df_test):
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            errors='ignore', inplace=True)

label_col = 'class2'
# only keep cat cols that exist in both
desired_cat = ['Protocol','Service','Conn_state']
cat_cols = [c for c in desired_cat if c in df_train.columns and c in df_test.columns]
# numeric features = all except label + cat_cols
feature_cols = [c for c in df_train.columns if c not in [label_col] + cat_cols]
# drop any numeric feature with no non-NaN in TRAIN
feature_cols = [c for c in feature_cols if df_train[c].notna().any()]

for c in feature_cols:
    df_train[c] = pd.to_numeric(df_train[c], errors='coerce')
    df_test [c] = pd.to_numeric(df_test [c], errors='coerce')
num_imputer = SimpleImputer(strategy='median')
X_train_num = num_imputer.fit_transform(df_train[feature_cols])
X_test_num  = num_imputer.transform   (df_test [feature_cols])

if cat_cols:
    df_train[cat_cols] = df_train[cat_cols].fillna('missing')
    df_test [cat_cols] = df_test [cat_cols].fillna('missing')
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(df_train[cat_cols])
    X_test_cat  = ohe.transform   (df_test [cat_cols])
else:
    X_train_cat = np.empty((len(df_train), 0))
    X_test_cat  = np.empty((len(df_test),  0))

X_train = np.hstack([X_train_num, X_train_cat])
X_test  = np.hstack([X_test_num , X_test_cat ])

le      = LabelEncoder()
y_train = le.fit_transform(df_train[label_col].fillna('missing'))
y_test  = le.transform   (df_test [label_col].fillna('missing'))
print("class2 mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test   = scaler.transform   (X_test)

# reshape for Conv1D/TCN input
seq_len    = X_train.shape[1]
feat_dim   = 1
X_train_seq = X_train.reshape(-1, seq_len, feat_dim)
X_test_seq  = X_test.reshape(-1, seq_len, feat_dim)
num_classes = len(le.classes_)

# BUILD TEACHER MODEL
def build_teacher():
    inp = layers.Input((seq_len, feat_dim))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    for rate in [1,2]:
        conv = layers.Conv1D(64,3,padding='causal',
                             dilation_rate=rate,activation='relu')(x)
        conv = layers.BatchNormalization()(conv)
        x    = layers.Add()([x, conv])
    scores  = layers.Dense(1, activation='tanh')(x)
    weights = layers.Softmax(axis=1)(scores)
    weighted= layers.Multiply()([x, weights])
    ctx     = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(weighted)
    out     = layers.Dense(num_classes, activation='softmax')(ctx)
    model   = models.Model(inp, out, name='Teacher')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')])
    return model

teacher = build_teacher()
mem_before = get_mem_mb()
t0 = time.time()
teacher.fit(
    X_train_seq, y_train,
    validation_split=0.1,
    epochs=1, batch_size=256, verbose=1
)
t1 = time.time()
mem_after = get_mem_mb()
print(f"Teacher training time: {t1 - t0:.2f} s")
print(f"Teacher memory RSS: before {mem_before:.1f} MB → after {mem_after:.1f} MB")

mem_before_inf = get_mem_mb()
t0_inf = time.time()
te_loss, te_acc = teacher.evaluate(X_test_seq, y_test, verbose=0)
t1_inf = time.time()
mem_after_inf = get_mem_mb()
print(f"Teacher inference time (evaluate): {t1_inf - t0_inf:.2f} s")
print(f"Teacher memory RSS: before inf {mem_before_inf:.1f} MB → after inf {mem_after_inf:.1f} MB")
print(f"Teacher eval loss={te_loss:.4f}, acc={te_acc:.4f}")

# GENERATE SOFT LABELS
T            = 10.0
train_logits = teacher.predict(X_train_seq, batch_size=512)
soft_train   = tf.nn.softmax(train_logits / T, axis=1)
test_logits  = teacher.predict(X_test_seq, batch_size=512)
soft_test    = tf.nn.softmax(test_logits / T, axis=1)

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_seq, y_train, soft_train)
).map(lambda x,y,s: (x,(y,s))).cache()\
 .shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)

val_ds   = tf.data.Dataset.from_tensor_slices(
    (X_test_seq, y_test, soft_test)
).map(lambda x,y,s: (x,(y,s))).batch(256).prefetch(tf.data.AUTOTUNE)

# Define Distiller & Student Model
class Distiller(models.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker       = tf.keras.metrics.Mean("loss")
        self.distill_tracker    = tf.keras.metrics.Mean("distill_loss")
        self.accuracy_tracker   = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.distill_tracker, self.accuracy_tracker]

    def compile(self, optimizer, student_loss_fn,
                distill_loss_fn, alpha=0.1, temperature=10):
        super().compile(optimizer=optimizer)
        self.student_loss_fn  = student_loss_fn
        self.distill_loss_fn  = distill_loss_fn
        self.alpha            = alpha
        self.temperature      = temperature

    def train_step(self, data):
        x,(y_true,y_soft)=data
        with tf.GradientTape() as tape:
            yp = self.student(x, training=True)
            yt = self.teacher(x, training=False)
            loss_h = self.student_loss_fn(y_true, yp)
            loss_s = self.distill_loss_fn(
                tf.nn.softmax(yt/self.temperature,axis=1),
                tf.nn.softmax(yp/self.temperature,axis=1))
            loss   = self.alpha*loss_h + (1-self.alpha)*loss_s
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.student.trainable_variables))
        self.loss_tracker.update_state(loss_h)
        self.distill_tracker.update_state(loss_s)
        self.accuracy_tracker.update_state(y_true, yp)
        return {"loss":self.loss_tracker.result(),
                "distill_loss":self.distill_tracker.result(),
                "accuracy":self.accuracy_tracker.result()}

    def test_step(self,data):
        x,(y_true,_) = data
        yp = self.student(x, training=False)
        loss_h = self.student_loss_fn(y_true, yp)
        self.loss_tracker.update_state(loss_h)
        self.accuracy_tracker.update_state(y_true, yp)
        return {"loss":self.loss_tracker.result(),
                "accuracy":self.accuracy_tracker.result()}

def build_student():
    inp = layers.Input((seq_len,feat_dim))
    x   = layers.GRU(64,implementation=2)(inp)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inp,out,name='Student_GRU')

student   = build_student()
distiller = Distiller(student, teacher)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    distill_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=T
)

mem_before_s = get_mem_mb()
t0_s = time.time()
distiller.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
t1_s = time.time()
mem_after_s = get_mem_mb()
print(f"Student distill‐training time: {t1_s - t0_s:.2f} s")
print(f"Student memory RSS: before {mem_before_s:.1f} MB → after {mem_after_s:.1f} MB")


mem_before_inf_s = get_mem_mb()
t0_inf_s = time.time()
y_prob = student.predict(X_test_seq, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
t1_inf_s = time.time()
mem_after_inf_s = get_mem_mb()
print(f"Student inference time (predict): {t1_inf_s - t0_inf_s:.2f} s")
print(f"Student memory RSS: before inf {mem_before_inf_s:.1f} MB → after inf {mem_after_inf_s:.1f} MB")

from sklearn.metrics import classification_report, precision_recall_fscore_support
print(classification_report(y_test, y_pred,
      target_names=le.classes_, digits=4))
p,r,f1,_ = precision_recall_fscore_support(
    y_test, y_pred, average='macro')
print(f"Macro Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

# FINAL EVALUATION
y_prob = student.predict(X_test_seq, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

from sklearn.metrics import classification_report, precision_recall_fscore_support
print(classification_report(y_test, y_pred,
      target_names=le.classes_, digits=4))
p,r,f1,_ = precision_recall_fscore_support(
    y_test, y_pred, average='macro')
print(f"Macro Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

# 12. STUDENT INFERENCE WITH RAM & TIME MEASUREMENT

mem_inf0 = proc.memory_info().rss / (1024**2)

start_inf = time.time()
y_prob_inf = student.predict(X_test_seq, batch_size=256, verbose=0)
inf_time = time.time() - start_inf

mem_inf1 = proc.memory_info().rss / (1024**2)

n_samples = X_test_seq.shape[0]
print(f"Inference time on test set: {inf_time:.4f}s for {n_samples} samples, "
      f"avg {inf_time/n_samples*1000:.4f} ms/sample")
print(f"Inference RAM Δ: {mem_inf1 - mem_inf0:.4f} MB")


