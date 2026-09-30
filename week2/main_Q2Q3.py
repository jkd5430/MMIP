# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:43:00 2026

@author: u1021
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression

# %%


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_PATH = r"C:\Users\u1021\Downloads\MMIP\week2\data\UCI_Credit_Card.csv"
TARGET = "default.payment.next.month"
df = pd.read_csv(DATA_PATH)


print("Dataset shape:", df.shape)
print("Target distribution:")
print(df[TARGET].value_counts(normalize=True))

X = df.drop(columns=["ID", TARGET])
y = df[TARGET].astype(np.float32).values

# 80% Training / 20% Validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
print("\nTrain shape:", X_train_scaled.shape)
print("Validation shape:", X_val_scaled.shape)

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
# Hidden layers:
# 23 input features -> 64 -> 32 -> 1
# Activation: ReLU
# Output: Sigmoid
# Optimizer: Adam
# Loss: Binary Crossentropy

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),
    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
)
model.summary()
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# 找 Validation Loss 最小的位置
best_epoch = np.argmin(val_loss) + 1
min_val_loss = np.min(val_loss)

print("\n========== Loss ==========")
print(f"Final Training Loss   : {train_loss[-1]:.4f}")
print(f"Final Validation Loss : {val_loss[-1]:.4f}")
print(f"Minimum Validation Loss: {min_val_loss:.4f}")
print(f"Best Epoch              : {best_epoch}")

train_probs = model.predict(
    X_train_scaled,
    verbose=0
).reshape(-1)

train_pred = (
    train_probs >= 0.5
).astype(int)

val_probs = model.predict(
    X_val_scaled,
    verbose=0
).reshape(-1)

val_pred = (
    val_probs >= 0.5
).astype(int)
train_accuracy = accuracy_score(
    y_train,
    train_pred
)
train_precision = precision_score(
    y_train,
    train_pred
)

train_recall = recall_score(
    y_train,
    train_pred
)

train_f1 = f1_score(
    y_train,
    train_pred
)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

val_precision = precision_score(
    y_val,
    val_pred
)

val_recall = recall_score(
    y_val,
    val_pred
)

val_f1 = f1_score(
    y_val,
    val_pred
)
print("\n========== Training Metrics ==========")
print(f"Accuracy  : {train_accuracy:.4f}")
print(f"Precision : {train_precision:.4f}")
print(f"Recall    : {train_recall:.4f}")
print(f"F1-Score  : {train_f1:.4f}")

print("\n========== Validation Metrics ==========")
print(f"Accuracy  : {val_accuracy:.4f}")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1-Score  : {val_f1:.4f}")
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, EPOCHS + 1),
    train_loss,
    label="Training Loss"
)
plt.plot(
    range(1, EPOCHS + 1),
    val_loss,
    label="Validation Loss"
)
plt.scatter(
    min_epoch,
    min_val_loss,
    s=80,
    label=f"Min Val Loss = {min_val_loss:.4f}"
)
plt.text(
    min_epoch,
    min_val_loss,
    f"({min_epoch}, {min_val_loss:.4f})",
    fontsize=10,
    ha="left",
    va="bottom"
)
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

auc = roc_auc_score(y_val, val_probs)
print(f"Validation ROC-AUC : {auc:.4f}")

# 展示 Validation Dataset 的一筆 Sample
sample_idx = 0

original_index = X_val.index[sample_idx]
sample_original = df.loc[original_index]
sample_scaled = X_val_scaled[sample_idx:sample_idx + 1]
sample_probability = model.predict(
    sample_scaled,
    verbose=0
)[0][0]
sample_prediction = int(sample_probability >= 0.5)
sample_truth = int(y_val[sample_idx])
print("\n========== One Validation Sample ==========")
print("Original row:")
print(sample_original)
print(f"\nTrue label = {sample_truth}")
print(
    "Predicted default probability = "
    f"{sample_probability:.4f}"
)
print(f"Predicted class = {sample_prediction}")

# %%

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
# Hidden layers:
# 23 input features -> 128 -> 64 -> 32 -> 1
# Activation: ReLU
# Output: Sigmoid
# Optimizer: Adam
# Loss: Binary Crossentropy

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),

    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),

    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),

    tf.keras.layers.Dropout(0.10),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
)
model.summary()
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# 找 Validation Loss 最小的位置
best_epoch = np.argmin(val_loss) + 1
min_val_loss = np.min(val_loss)

print("\n========== Loss ==========")
print(f"Final Training Loss   : {train_loss[-1]:.4f}")
print(f"Final Validation Loss : {val_loss[-1]:.4f}")
print(f"Minimum Validation Loss: {min_val_loss:.4f}")
print(f"Best Epoch              : {best_epoch}")

train_probs = model.predict(
    X_train_scaled,
    verbose=0
).reshape(-1)

train_pred = (
    train_probs >= 0.5
).astype(int)

val_probs = model.predict(
    X_val_scaled,
    verbose=0
).reshape(-1)

val_pred = (
    val_probs >= 0.5
).astype(int)
train_accuracy = accuracy_score(
    y_train,
    train_pred
)
train_precision = precision_score(
    y_train,
    train_pred
)

train_recall = recall_score(
    y_train,
    train_pred
)

train_f1 = f1_score(
    y_train,
    train_pred
)


# --------------------------
# 5. Validation Metrics
# --------------------------
val_accuracy = accuracy_score(
    y_val,
    val_pred
)

val_precision = precision_score(
    y_val,
    val_pred
)

val_recall = recall_score(
    y_val,
    val_pred
)

val_f1 = f1_score(
    y_val,
    val_pred
)

print("\n========== Training Metrics ==========")
print(f"Accuracy  : {train_accuracy:.4f}")
print(f"Precision : {train_precision:.4f}")
print(f"Recall    : {train_recall:.4f}")
print(f"F1-Score  : {train_f1:.4f}")

print("\n========== Validation Metrics ==========")
print(f"Accuracy  : {val_accuracy:.4f}")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1-Score  : {val_f1:.4f}")
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, EPOCHS + 1),
    train_loss,
    label="Training Loss"
)
plt.plot(
    range(1, EPOCHS + 1),
    val_loss,
    label="Validation Loss"
)
plt.scatter(
    min_epoch,
    min_val_loss,
    s=80,
    label=f"Min Val Loss = {min_val_loss:.4f}"
)
plt.text(
    min_epoch,
    min_val_loss,
    f"({min_epoch}, {min_val_loss:.4f})",
    fontsize=10,
    ha="left",
    va="bottom"
)
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

auc = roc_auc_score(y_val, val_probs)
print(f"Validation ROC-AUC : {auc:.4f}")
# %%Q3

fpr, tpr, thresholds = roc_curve(
    y_val,
    val_probs
)
roc_auc = roc_auc_score(
    y_val,
    val_probs
)

print(f"\nAUC = {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
# ROC Curve
plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"MLP ROC Curve (AUC = {roc_auc:.4f})"
)

# Random classifier baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier (AUC = 0.5)"
)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve of MLP on Validation Dataset")

plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
MLP_val_probs = model.predict(
    X_val_scaled,
    verbose=0
).reshape(-1)

MLP_fpr, MLP_tpr, MLP_thresholds = roc_curve(
    y_val,
    MLP_val_probs
)

MLP_auc = roc_auc_score(
    y_val,
    MLP_val_probs
)

print(f"MLP AUC = {MLP_auc:.4f}")

LG_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
LG_model.fit(
    X_train_scaled,
    y_train
)
LG_val_probs = LG_model.predict_proba(
    X_val_scaled
)[:, 1]
print("\nFirst 10 Logistic Regression Probabilities:")
print(LG_val_probs[:10])
LG_fpr, LG_tpr, LG_thresholds = roc_curve(
    y_val,
    LG_val_probs
)
LG_auc = roc_auc_score(
    y_val,
    LG_val_probs
)
print(f"\nLogistic Regression AUC = {LG_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(
    MLP_fpr,
    MLP_tpr,
    linewidth=2,
    label=f"MLP (AUC = {MLP_auc:.4f})"
)
plt.plot(
    LG_fpr,
    LG_tpr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {LG_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier (AUC = 0.5)"
)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

plt.title(
    "ROC Curve Comparison: MLP vs Logistic Regression"
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()











