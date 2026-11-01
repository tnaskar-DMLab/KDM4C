# ==============================================================
# EXTERNAL VALIDATION OF OPTIMIZED kNN MODEL
#
# Development dataset:
#   normalized_descriptors.xlsx
#
# External validation dataset:
#   ex_val_set.xlsx
#
# Optimized kNN:
#   k = 8
#   weights = uniform
#   p = 1 (Manhattan distance)
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ==============================================================
# 1. INPUT FILES
# ==============================================================

development_file = "normalized_descriptors.xlsx"
external_file = "ex_val_set.xlsx"


if not os.path.exists(development_file):
    raise FileNotFoundError(
        f"Development dataset not found: {development_file}"
    )

if not os.path.exists(external_file):
    raise FileNotFoundError(
        f"External validation dataset not found: {external_file}"
    )


print("=" * 70)
print("EXTERNAL VALIDATION OF OPTIMIZED kNN MODEL")
print("=" * 70)

print(f"Development dataset : {development_file}")
print(f"External dataset    : {external_file}")


# ==============================================================
# 2. SELECTED 14 DESCRIPTORS
# ==============================================================

features = [
    "BCUT2D_MWHI",
    "PEOE_VSA13",
    "SlogP_VSA7",
    "PEOE_VSA5",
    "fr_Ar_NH",
    "n6HRing",
    "ATSC2i",
    "MIC1",
    "AATSC7dv",
    "SlogP_VSA4",
    "ECIndex",
    "nAcid",
    "PEOE_VSA4",
    "VSA_EState9"
]


# ==============================================================
# 3. LOAD DEVELOPMENT AND EXTERNAL DATASETS
# ==============================================================

df_train = pd.read_excel(
    development_file
)

df_ext = pd.read_excel(
    external_file
)


print("\nDataset dimensions:")
print("Development :", df_train.shape)
print("External    :", df_ext.shape)


# ==============================================================
# 4. CHECK REQUIRED COLUMNS
# ==============================================================

for dataset_name, dataframe in [
    ("Development", df_train),
    ("External", df_ext)
]:

    missing_features = [
        f for f in features
        if f not in dataframe.columns
    ]

    if missing_features:
        raise KeyError(
            f"{dataset_name} dataset is missing descriptors:\n"
            f"{missing_features}"
        )

    if "bioactivity_class" not in dataframe.columns:
        raise KeyError(
            f"'bioactivity_class' not found in {dataset_name} dataset."
        )


# ==============================================================
# 5. PREPARE DEVELOPMENT DATA
# ==============================================================

X_train = df_train[
    features
].copy()


y_train = df_train[
    "bioactivity_class"
].map(
    lambda x:
    1 if str(x).strip().lower() == "active"
    else 0
)


print("\nDevelopment-set class distribution:")
print(
    y_train.value_counts()
)


# ==============================================================
# 6. PREPARE EXTERNAL DATA
# ==============================================================

X_ext = df_ext[
    features
].copy()


y_ext = df_ext[
    "bioactivity_class"
].map(
    lambda x:
    1 if str(x).strip().lower() == "active"
    else 0
)


print("\nExternal-set class distribution:")
print(
    y_ext.value_counts()
)


# ==============================================================
# 7. IMPUTATION
#
# IMPORTANT:
# Fit imputer ONLY on development dataset.
# External dataset must not influence imputation.
# ==============================================================

imputer = SimpleImputer(
    strategy="mean"
)


X_train_imputed = imputer.fit_transform(
    X_train
)


X_ext_imputed = imputer.transform(
    X_ext
)


# ==============================================================
# 8. STANDARDIZATION
#
# IMPORTANT:
# Fit scaler ONLY on development dataset.
# ==============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_imputed
)


X_ext_scaled = scaler.transform(
    X_ext_imputed
)


# ==============================================================
# 9. TRAIN FINAL OPTIMIZED kNN MODEL
#
# Optimized parameters:
#   k = 8
#   uniform weighting
#   Manhattan distance
# ==============================================================

knn_model = KNeighborsClassifier(
    n_neighbors=8,
    weights="uniform",
    algorithm="auto",
    metric="minkowski",
    p=1,
    n_jobs=-1
)


knn_model.fit(
    X_train_scaled,
    y_train
)


print("\nOptimized model:")
print("k               = 8")
print("weights         = uniform")
print("distance metric = Manhattan")
print("p               = 1")


# ==============================================================
# 10. EXTERNAL PREDICTIONS
# ==============================================================

y_ext_pred = knn_model.predict(
    X_ext_scaled
)


y_ext_proba = knn_model.predict_proba(
    X_ext_scaled
)[:, 1]


# ==============================================================
# 11. CONFUSION MATRIX
# ==============================================================

cm_ext = confusion_matrix(
    y_ext,
    y_ext_pred,
    labels=[0, 1]
)


tn, fp, fn, tp = cm_ext.ravel()


# ==============================================================
# 12. EXTERNAL VALIDATION METRICS
# ==============================================================

accuracy = accuracy_score(
    y_ext,
    y_ext_pred
)


balanced_acc = balanced_accuracy_score(
    y_ext,
    y_ext_pred
)


sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)


precision = precision_score(
    y_ext,
    y_ext_pred,
    zero_division=0
)


recall = recall_score(
    y_ext,
    y_ext_pred,
    zero_division=0
)


f1 = f1_score(
    y_ext,
    y_ext_pred,
    zero_division=0
)


mcc = matthews_corrcoef(
    y_ext,
    y_ext_pred
)


# ROC-AUC requires both classes
if y_ext.nunique() == 2:

    roc_auc = roc_auc_score(
        y_ext,
        y_ext_proba
    )

else:

    roc_auc = np.nan


# ==============================================================
# 13. PRINT RESULTS
# ==============================================================

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 70)


print("\nConfusion Matrix:")

print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")


print("\nPerformance:")

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_acc:.4f}"
)

print(
    f"Sensitivity       : {sensitivity:.4f}"
)

print(
    f"Specificity       : {specificity:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print(
    f"MCC               : {mcc:.4f}"
)


if not np.isnan(roc_auc):

    print(
        f"ROC AUC           : {roc_auc:.4f}"
    )

else:

    print(
        "ROC AUC           : Not calculated "
        "(external set contains only one class)"
    )


print("\nClassification Report:\n")

print(
    classification_report(
        y_ext,
        y_ext_pred,
        target_names=[
            "Inactive",
            "Active"
        ],
        digits=4,
        zero_division=0
    )
)


# ==============================================================
# 14. SAVE METRICS TABLE
# ==============================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "TN",
        "TP",
        "FN",
        "FP",
        "Sensitivity",
        "Specificity",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
        "ROC AUC"
    ],

    "Value": [
        tn,
        tp,
        fn,
        fp,
        sensitivity,
        specificity,
        accuracy,
        balanced_acc,
        precision,
        recall,
        f1,
        mcc,
        roc_auc
    ]

})


metrics_df.to_excel(
    "knn_external_validation_metrics.xlsx",
    index=False
)


# ==============================================================
# 15. CONFUSION MATRIX PLOT
# ==============================================================

plt.figure(
    figsize=(7, 6)
)


sns.heatmap(
    cm_ext,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    square=True,
    xticklabels=[
        "Inactive",
        "Active"
    ],
    yticklabels=[
        "Inactive",
        "Active"
    ],
    annot_kws={
        "size": 22
    }
)


plt.xlabel(
    "Predicted label",
    fontsize=16
)

plt.ylabel(
    "True label",
    fontsize=16
)

plt.title(
    "Confusion Matrix - External Validation",
    fontsize=18,
    fontweight="bold"
)


plt.xticks(
    fontsize=14
)

plt.yticks(
    fontsize=14
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix_knn_external.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()


# ==============================================================
# 16. ROC CURVE
# ==============================================================

if y_ext.nunique() == 2:

    fpr_ext, tpr_ext, thresholds_ext = roc_curve(
        y_ext,
        y_ext_proba
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.plot(
        fpr_ext,
        tpr_ext,
        linewidth=2.5,
        label=f"Optimized kNN (AUC = {roc_auc:.3f})"
    )


    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1.8,
        label="Random classifier"
    )


    plt.xlabel(
        "False Positive Rate",
        fontsize=16
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=16
    )

    plt.title(
        "ROC Curve - External Validation",
        fontsize=18,
        fontweight="bold"
    )


    plt.xticks(
        fontsize=14
    )

    plt.yticks(
        fontsize=14
    )


    plt.legend(
        loc="lower right",
        fontsize=13
    )


    plt.tight_layout()


    plt.savefig(
        "roc_auc_curve_knn_external.png",
        dpi=600,
        bbox_inches="tight"
    )


    plt.close()


# ==============================================================
# 17. SAVE EXTERNAL PREDICTIONS
# ==============================================================

output_data = {}


# Keep Compound ID if available
if "Compound_ID" in df_ext.columns:

    output_data[
        "Compound_ID"
    ] = df_ext[
        "Compound_ID"
    ].values


# Keep SMILES if available
if "SMILES" in df_ext.columns:

    output_data[
        "SMILES"
    ] = df_ext[
        "SMILES"
    ].values


output_data[
    "Actual"
] = y_ext.values


output_data[
    "Predicted"
] = y_ext_pred


output_data[
    "Probability_Active"
] = y_ext_proba


output_ext_df = pd.DataFrame(
    output_data
)


output_ext_df[
    "Actual_Class"
] = output_ext_df[
    "Actual"
].map(
    {
        0: "Inactive",
        1: "Active"
    }
)


output_ext_df[
    "Predicted_Class"
] = output_ext_df[
    "Predicted"
].map(
    {
        0: "Inactive",
        1: "Active"
    }
)


output_ext_df[
    "Correct_Prediction"
] = (
    output_ext_df[
        "Actual"
    ]
    ==
    output_ext_df[
        "Predicted"
    ]
)


output_ext_df.to_excel(
    "knn_predictions_external.xlsx",
    index=False
)


# ==============================================================
# 18. SAVE FINAL MODEL
# ==============================================================

joblib.dump(
    knn_model,
    "knn_classifier_external_final.pkl"
)


joblib.dump(
    scaler,
    "standard_scaler_external_final.pkl"
)


joblib.dump(
    imputer,
    "mean_imputer_external_final.pkl"
)


# ==============================================================
# 19. SAVE MODEL SETTINGS
# ==============================================================

settings_df = pd.DataFrame({

    "Parameter": [
        "Number of descriptors",
        "n_neighbors",
        "weights",
        "Distance metric",
        "Minkowski p",
        "Development compounds",
        "External compounds"
    ],

    "Value": [
        len(features),
        8,
        "uniform",
        "Manhattan",
        1,
        len(df_train),
        len(df_ext)
    ]

})


settings_df.to_excel(
    "knn_external_model_settings.xlsx",
    index=False
)


# ==============================================================
# 20. FINISH
# ==============================================================

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 70)


print("\nFiles generated:")

print(
    " - knn_external_validation_metrics.xlsx"
)

print(
    " - knn_predictions_external.xlsx"
)

print(
    " - confusion_matrix_knn_external.png"
)

if y_ext.nunique() == 2:

    print(
        " - roc_auc_curve_knn_external.png"
    )

print(
    " - knn_external_model_settings.xlsx"
)

print(
    " - knn_classifier_external_final.pkl"
)

print(
    " - standard_scaler_external_final.pkl"
)

print(
    " - mean_imputer_external_final.pkl"
)