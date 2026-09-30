# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:43:00 2026

@author: u1021
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

file_name = r"C:\Users\u1021\Downloads\MMIP\week2\data\image_full_fast035_171_205_all_image_metrics.csv"

df = pd.read_csv(file_name)

print("Dataset shape:", df.shape)
print("\n前 5 筆資料：")
display(df.head())
print("\nLabel 分布：")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

# 排除 patient、patient_uid、filename、path 等識別資訊，
FEATURES = [
    "zero_pct",
    "dark_le5_pct",
    "nonzero_pct",
    "roi_gt5_pct",

    "full_mean",
    "full_std",
    "full_entropy",
    "full_snr",
    "full_cv",
    "full_enl",

    "roi_mean",
    "roi_std",
    "roi_entropy",
    "roi_snr",
    "roi_cv",
    "roi_enl",

    "sobel_mean_128roi",
    "sobel_std_128roi",
    "laplacian_var_128roi",
    "gradient_mean_128roi",
    "gradient_p95_128roi",
    "tenengrad_mean_128roi",
    "brenner_mean_128roi",
    "canny_edge_density_128roi",

    "fft_high_freq_ratio_r025_128roi",
    "fft_high_freq_ratio_r040_128roi",

    "glcm_contrast_128roi",
    "glcm_dissimilarity_128roi",
    "glcm_homogeneity_128roi",
    "glcm_ASM_128roi",
    "glcm_energy_128roi",
    "glcm_correlation_128roi"
]
TARGET = "label"
print("\n使用的 Feature 數量：", len(FEATURES))
print("\nFeatures:")
for feature in FEATURES:
    print("-", feature)

# 先以 patient 為單位切分，避免同一病人同時出現在 Training 和 Validation。
patient_table = (
    df[["patient_uid", TARGET]]
    .drop_duplicates()
    .reset_index(drop=True)
)
train_patients, val_patients = train_test_split(
    patient_table["patient_uid"],
    test_size=0.20,
    random_state=42,
    stratify=patient_table[TARGET]
)
train_df = df[df["patient_uid"].isin(train_patients)].copy()
val_df = df[df["patient_uid"].isin(val_patients)].copy()
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

print("\n==============================")
print("Training / Validation Split")
print("==============================")
print("Training samples :", len(X_train))
print("Validation samples:", len(X_val))
print("\nTraining Label Distribution:")
print(y_train.value_counts())
print("\nValidation Label Distribution:")
print(y_val.value_counts())


# 缺失值使用 median 補值 StandardScaler 標準化 Logistic Regression 分類

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_val)[:, 1]

probability_result = pd.DataFrame({
    "Actual_Label": y_val.values,
    "Probability_Class_1": y_prob
})

print("\n==============================")
print("Classification Probability")
print("==============================")

display(probability_result.head(20))

auc = roc_auc_score(y_val, y_prob)
print("ROC-AUC =", round(auc, 4))

def evaluate_threshold(y_true, y_probability, threshold):

    # probability >= threshold -> class 1
    # probability < threshold  -> class 0

    y_pred = (y_probability >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return y_pred, cm, accuracy, precision, recall, f1

initial_threshold = 0.50
(
    y_pred_initial,
    cm_initial,
    accuracy_initial,
    precision_initial,
    recall_initial,
    f1_initial
) = evaluate_threshold(
    y_val,
    y_prob,
    initial_threshold
)
print("\n")
print("=" * 60)
print("Initial Threshold =", initial_threshold)
print("=" * 60)
print("\nConfusion Matrix:")
print(cm_initial)
print("\nPerformance:")
print(f"Accuracy  : {accuracy_initial:.4f}")
print(f"Precision : {precision_initial:.4f}")
print(f"Recall    : {recall_initial:.4f}")
print(f"F1-Score  : {f1_initial:.4f}")


# 畫 Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_initial,
    display_labels=[0, 1]
)
disp.plot()
plt.title(
    f"Confusion Matrix - Threshold = {initial_threshold:.2f}"
)
plt.show()



# 嘗試 Threshold 0.10 ~ 0.90 用F1-Score選擇Threshold
thresholds = np.arange(
    0.10,
    0.91,
    0.01
)
threshold_results = []
for threshold in thresholds:

    (
        y_pred_temp,
        cm_temp,
        accuracy_temp,
        precision_temp,
        recall_temp,
        f1_temp
    ) = evaluate_threshold(
        y_val,
        y_prob,
        threshold
    )

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_temp,
        "Precision": precision_temp,
        "Recall": recall_temp,
        "F1": f1_temp
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\nThreshold Results:")
display(threshold_df)
best_index = threshold_df["F1"].idxmax()
best_threshold = threshold_df.loc[
    best_index,
    "Threshold"
]
print("\n")
print("=" * 60)
print("Threshold Fine-Tuning Result")
print("=" * 60)
print(
    f"Best Threshold based on F1-Score = "
    f"{best_threshold:.2f}"
)
plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Accuracy"],
    label="Accuracy"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1-Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Classification Metrics vs Threshold"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()
(
    y_pred_best,
    cm_best,
    accuracy_best,
    precision_best,
    recall_best,
    f1_best
) = evaluate_threshold(
    y_val,
    y_prob,
    best_threshold
)


print("\n")
print("=" * 60)
print(
    f"Fine-Tuned Threshold = "
    f"{best_threshold:.2f}"
)
print("=" * 60)

print("\nConfusion Matrix:")
print(cm_best)

print("\nPerformance:")
print(f"Accuracy  : {accuracy_best:.4f}")
print(f"Precision : {precision_best:.4f}")
print(f"Recall    : {recall_best:.4f}")
print(f"F1-Score  : {f1_best:.4f}")


# 畫新的 Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_best,
    display_labels=[0, 1]
)

disp.plot()

plt.title(
    f"Confusion Matrix - Fine-Tuned Threshold = "
    f"{best_threshold:.2f}"
)

plt.show()

comparison = pd.DataFrame({
    "Method": [
        "Initial Threshold",
        "Fine-Tuned Threshold"
    ],

    "Threshold": [
        initial_threshold,
        best_threshold
    ],

    "Accuracy": [
        accuracy_initial,
        accuracy_best
    ],

    "Precision": [
        precision_initial,
        precision_best
    ],

    "Recall": [
        recall_initial,
        recall_best
    ],

    "F1-Score": [
        f1_initial,
        f1_best
    ]
})
print("\n")
print("=" * 60)
print("Before vs After Threshold Tuning")
print("=" * 60)

display(
    comparison.round(4)
)
# %% 進階 rF
#Random Forest不需要做StandardScaler
model_rf = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    )
])

model_rf.fit(X_train, y_train)
y_prob_rf = model_rf.predict_proba(X_val)[:, 1]
probability_result_rf = pd.DataFrame({
    "Actual_Label": y_val.values,
    "Probability_Class_1": y_prob_rf
})
thresholds_rf = np.arange(
    0.10,
    0.91,
    0.01
)
threshold_results_rf = []
for threshold in thresholds_rf:
    (
        y_pred_temp,
        cm_temp,
        accuracy_temp,
        precision_temp,
        recall_temp,
        f1_temp
    ) = evaluate_threshold(
        y_val,
        y_prob_rf,
        threshold
    )
    threshold_results_rf.append({
        "Threshold": threshold,
        "Accuracy": accuracy_temp,
        "Precision": precision_temp,
        "Recall": recall_temp,
        "F1": f1_temp
    })
threshold_df_rf = pd.DataFrame(
    threshold_results_rf
)
print("\nRandom Forest Threshold Results:")
display(threshold_df_rf)
best_index_rf = threshold_df_rf["F1"].idxmax()
best_threshold_rf = threshold_df_rf.loc[
    best_index_rf,
    "Threshold"
]
print("\n")
print("=" * 60)
print("Random Forest Threshold Fine-Tuning Result")
print("=" * 60)
print(
    f"Best Threshold based on F1-Score = "
    f"{best_threshold_rf:.2f}"
)
(
    y_pred_rf,
    cm_rf,
    accuracy_rf,
    precision_rf,
    recall_rf,
    f1_rf
) = evaluate_threshold(
    y_val,
    y_prob_rf,
    best_threshold_rf
)


print("\n")
print("=" * 60)
print("Random Forest Final Result")
print("=" * 60)

print(
    f"Fine-Tuned Threshold = "
    f"{best_threshold_rf:.2f}"
)

print("\nConfusion Matrix:")
print(cm_rf)

print("\nPerformance:")
print(f"Accuracy  : {accuracy_rf:.4f}")
print(f"Precision : {precision_rf:.4f}")
print(f"Recall    : {recall_rf:.4f}")
print(f"F1-Score  : {f1_rf:.4f}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=[0, 1]
)

disp.plot()

plt.title(
    f"Random Forest Confusion Matrix\n"
    f"Threshold = {best_threshold_rf:.2f}"
)
plt.show()

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Best Threshold": [
        best_threshold,
        best_threshold_rf
    ],

    "Accuracy": [
        accuracy_best,
        accuracy_rf
    ],

    "Precision": [
        precision_best,
        precision_rf
    ],

    "Recall": [
        recall_best,
        recall_rf
    ],

    "F1-Score": [
        f1_best,
        f1_rf
    ]
})


print("\n")
print("=" * 70)
print("Logistic Regression vs Random Forest")
print("=" * 70)

display(comparison.round(4))

