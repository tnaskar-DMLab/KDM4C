import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime


# ==========================================================
# 1. OUTPUT DIRECTORY
# ==========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = os.path.join(
    os.getcwd(),
    f"knn_loo_cv_results_{timestamp}"
)

os.makedirs(output_dir, exist_ok=True)

print(f"Output directory: {output_dir}")


# ==========================================================
# 2. LOAD TRAINING DATA ONLY
# ==========================================================

# IMPORTANT:
# This file should contain ONLY the 680 training compounds.
# The 170 independent test compounds must NOT be present here.

data_file = "train_set.xlsx"

if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Dataset file '{data_file}' not found in current directory."
    )

data = pd.read_excel(data_file)

print("\nDataset loaded successfully")
print(f"Original dataset shape: {data.shape}")


# ==========================================================
# 3. DESCRIPTORS
# ==========================================================

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

TARGET_COL = "bioactivity_class"


# ==========================================================
# 4. CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = features + [TARGET_COL]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ==========================================================
# 5. REMOVE ROWS WITH MISSING VALUES
# ==========================================================

before_drop = len(data)

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)

after_drop = len(data)

print(f"Rows before removing missing values: {before_drop}")
print(f"Rows after removing missing values:  {after_drop}")


# ==========================================================
# 6. PREPARE X AND y
# ==========================================================

X = data[features].to_numpy(dtype=float)

y_raw = data[TARGET_COL]


# ----------------------------------------------------------
# Handle target variable carefully
# We require:
# 0 = Inactive
# 1 = Active
# ----------------------------------------------------------

if (
    y_raw.dtype == "object"
    or str(y_raw.dtype).startswith("string")
):

    y_text = (
        y_raw.astype(str)
        .str.strip()
        .str.lower()
    )

    unique_labels = set(y_text.unique())

    if unique_labels.issubset({"active", "inactive"}):

        mapping = {
            "inactive": 0,
            "active": 1
        }

        y = y_text.map(mapping).to_numpy(dtype=int)

        print("\nTarget encoding:")
        print("0 = Inactive")
        print("1 = Active")

    else:
        raise ValueError(
            "Target column contains unexpected text labels. "
            f"Found: {sorted(unique_labels)}"
        )

else:

    y = y_raw.to_numpy(dtype=int)

    unique_labels = np.unique(y)

    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            "Numeric target must contain only 0 and 1. "
            f"Found: {unique_labels}"
        )


# ==========================================================
# 7. DATASET INFORMATION
# ==========================================================

n_samples = len(X)

print("\n" + "=" * 65)
print("TRAINING DATASET INFORMATION")
print("=" * 65)

print(f"Dataset shape: {X.shape}")
print(f"Number of descriptors: {X.shape[1]}")
print(f"Number of samples: {n_samples}")

class_counts = np.bincount(y, minlength=2)

print(f"Inactive compounds (0): {class_counts[0]}")
print(f"Active compounds   (1): {class_counts[1]}")


# Optional safety check
if n_samples != 680:
    print(
        f"\nWARNING: Expected 680 training compounds, "
        f"but found {n_samples}."
    )


# ==========================================================
# 8. FIXED KNN HYPERPARAMETERS
# ==========================================================

# These parameters must have been selected using
# TRAINING DATA ONLY, not the independent 170 compounds.

N_NEIGHBORS = 8
P_VALUE = 1
WEIGHTS = "uniform"


print("\nKNN hyperparameters:")
print(f"n_neighbors = {N_NEIGHBORS}")
print(f"p = {P_VALUE} (Manhattan distance)")
print(f"weights = {WEIGHTS}")


# ==========================================================
# 9. INITIALIZE LEAVE-ONE-OUT CV
# ==========================================================

loo = LeaveOneOut()

print("\n" + "=" * 65)
print("STARTING LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 65)

print(f"Total LOOCV iterations: {n_samples}")
print(
    "Each iteration: "
    f"{n_samples - 1} training compound(s) + 1 validation compound"
)


# ==========================================================
# 10. STORAGE
# ==========================================================

y_true = []
y_pred = []
y_proba = []

train_accuracies = []
left_out_indices = []


# ==========================================================
# 11. LEAKAGE-FREE LOOCV
# ==========================================================

for i, (train_index, test_index) in enumerate(loo.split(X)):

    if (i + 1) % 100 == 0 or (i + 1) == n_samples:
        print(
            f"Processing sample {i + 1}/{n_samples}"
        )

    # ------------------------------------------------------
    # Split RAW data first
    # ------------------------------------------------------

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]


    # ------------------------------------------------------
    # Fit scaler ONLY on current training fold
    #
    # This avoids information leakage from the left-out
    # validation compound.
    # ------------------------------------------------------

    scaler_fold = StandardScaler()

    X_train_scaled = scaler_fold.fit_transform(X_train)

    X_test_scaled = scaler_fold.transform(X_test)


    # ------------------------------------------------------
    # Define KNN model
    # ------------------------------------------------------

    model = KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights=WEIGHTS,
        p=P_VALUE,
        metric="minkowski",
        n_jobs=-1
    )


    # ------------------------------------------------------
    # Train using n-1 compounds
    # ------------------------------------------------------

    model.fit(
        X_train_scaled,
        y_train
    )


    # ------------------------------------------------------
    # Predict left-out compound
    # ------------------------------------------------------

    pred = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )


    # Find probability corresponding specifically to class 1
    active_class_index = np.where(
        model.classes_ == 1
    )[0][0]

    proba_active = probabilities[
        0,
        active_class_index
    ]


    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    y_true.append(
        int(y_test[0])
    )

    y_pred.append(
        int(pred[0])
    )

    y_proba.append(
        float(proba_active)
    )

    left_out_indices.append(
        int(test_index[0])
    )


    # ------------------------------------------------------
    # Training accuracy for this fold
    # ------------------------------------------------------

    train_pred = model.predict(
        X_train_scaled
    )

    train_acc = accuracy_score(
        y_train,
        train_pred
    )

    train_accuracies.append(
        train_acc
    )


# ==========================================================
# 12. CONVERT RESULTS TO NUMPY ARRAYS
# ==========================================================

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
y_proba = np.asarray(y_proba)
train_accuracies = np.asarray(train_accuracies)


# ==========================================================
# 13. CALCULATE PERFORMANCE METRICS
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_proba
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


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

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else np.nan
)

npv = (
    tn / (tn + fn)
    if (tn + fn) > 0
    else np.nan
)

f1 = (
    2 * precision * sensitivity /
    (precision + sensitivity)
    if (
        not np.isnan(precision)
        and not np.isnan(sensitivity)
        and (precision + sensitivity) > 0
    )
    else np.nan
)


# ==========================================================
# 14. PRINT RESULTS
# ==========================================================

print("\n" + "=" * 65)
print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
print("=" * 65)

print(f"Number of samples:       {n_samples}")
print(f"Accuracy:                {accuracy:.4f}")
print(f"Balanced Accuracy:       {balanced_acc:.4f}")
print(f"ROC AUC:                 {roc_auc:.4f}")
print(f"Sensitivity:             {sensitivity:.4f}")
print(f"Specificity:             {specificity:.4f}")
print(f"Precision:               {precision:.4f}")
print(f"Negative Predictive Val: {npv:.4f}")
print(f"F1 Score:                {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

classification_results = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=[
        "Inactive",
        "Active"
    ],
    digits=4,
    zero_division=0
)

print(classification_results)


# ==========================================================
# 15. SAVE INDIVIDUAL LOOCV PREDICTIONS
# ==========================================================

predictions_df = pd.DataFrame({

    "sample_index":
        left_out_indices,

    "actual":
        y_true,

    "predicted":
        y_pred,

    "probability_active":
        y_proba,

    "correct":
        (y_true == y_pred).astype(int)
})

predictions_file = os.path.join(
    output_dir,
    "loo_cv_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ==========================================================
# 16. SAVE METRICS SUMMARY
# ==========================================================

metrics_summary = {

    "Number_of_Samples":
        n_samples,

    "Inactive":
        int(class_counts[0]),

    "Active":
        int(class_counts[1]),

    "Accuracy":
        accuracy,

    "Balanced_Accuracy":
        balanced_acc,

    "ROC_AUC":
        roc_auc,

    "Sensitivity":
        sensitivity,

    "Specificity":
        specificity,

    "Precision":
        precision,

    "NPV":
        npv,

    "F1_Score":
        f1,

    "True_Negatives":
        int(tn),

    "False_Positives":
        int(fp),

    "False_Negatives":
        int(fn),

    "True_Positives":
        int(tp)
}

metrics_df = pd.DataFrame(
    [metrics_summary]
)

metrics_file = os.path.join(
    output_dir,
    "loo_cv_metrics_summary.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ==========================================================
# 17. CONFUSION MATRIX FIGURE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Inactive",
        "Active"
    ]
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "KNN Leave-One-Out Cross-Validation"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "loo_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 18. ROC CURVE
# ==========================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_proba
)

plt.figure(
    figsize=(6, 5)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"KNN LOOCV (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Leave-One-Out Cross-Validation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "loo_roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 19. PREDICTED PROBABILITY DISTRIBUTION
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

inactive_mask = (
    y_true == 0
)

active_mask = (
    y_true == 1
)

plt.hist(
    y_proba[inactive_mask],
    bins=20,
    alpha=0.6,
    label="Inactive"
)

plt.hist(
    y_proba[active_mask],
    bins=20,
    alpha=0.6,
    label="Active"
)

plt.xlabel(
    "Predicted Probability of Active Class"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "LOOCV Predicted Probability Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "loo_probability_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 20. TRAINING ACCURACY DISTRIBUTION
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    train_accuracies,
    bins=20,
    edgecolor="black"
)

plt.xlabel(
    "Training Accuracy"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Training Accuracy Across LOOCV Folds"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "loo_training_accuracy_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 21. TRAIN FINAL MODEL ON ALL 680 TRAINING COMPOUNDS
# ==========================================================

print("\n" + "=" * 65)
print("TRAINING FINAL MODEL ON COMPLETE TRAINING SET")
print("=" * 65)


# Fit scaler using ALL training compounds.
# This scaler will later be used to transform the independent
# 170-compound external test set.

final_scaler = StandardScaler()

X_scaled_final = final_scaler.fit_transform(
    X
)


final_model = KNeighborsClassifier(
    n_neighbors=N_NEIGHBORS,
    weights=WEIGHTS,
    p=P_VALUE,
    metric="minkowski",
    n_jobs=-1
)


final_model.fit(
    X_scaled_final,
    y
)


# ==========================================================
# 22. SAVE FINAL MODEL AND SCALER
# ==========================================================

model_file = os.path.join(
    output_dir,
    "knn_final_model_optimized.pkl"
)

scaler_file = os.path.join(
    output_dir,
    "standard_scaler.pkl"
)

joblib.dump(
    final_model,
    model_file
)

joblib.dump(
    final_scaler,
    scaler_file
)

print("Final KNN model saved.")
print("Final StandardScaler saved.")


# ==========================================================
# 23. SAVE DETAILED REPORT
# ==========================================================

report_file = os.path.join(
    output_dir,
    "detailed_summary_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "KNN LEAVE-ONE-OUT CROSS-VALIDATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Timestamp: {timestamp}\n"
    )

    f.write(
        f"Dataset: {data_file}\n"
    )

    f.write(
        f"Training compounds: {n_samples}\n"
    )

    f.write(
        f"Number of descriptors: {len(features)}\n"
    )

    f.write(
        f"Inactive compounds: {class_counts[0]}\n"
    )

    f.write(
        f"Active compounds: {class_counts[1]}\n\n"
    )


    f.write(
        "VALIDATION STRATEGY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Leave-One-Out Cross-Validation was performed only on "
        "the training dataset.\n"
    )

    f.write(
        "For each iteration, one compound was held out for "
        "validation and the model was trained on all remaining "
        "training compounds.\n"
    )

    f.write(
        "Feature standardization was performed independently "
        "within every LOOCV fold. The StandardScaler was fitted "
        "only on the training portion of each fold and then "
        "applied to the left-out compound.\n"
    )

    f.write(
        "The independent external test set was not used during "
        "LOOCV.\n\n"
    )


    f.write(
        "KNN HYPERPARAMETERS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"n_neighbors: {N_NEIGHBORS}\n"
    )

    f.write(
        f"p: {P_VALUE} (Manhattan distance)\n"
    )

    f.write(
        f"weights: {WEIGHTS}\n\n"
    )


    f.write(
        "LOOCV PERFORMANCE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: {balanced_acc:.4f}\n"
    )

    f.write(
        f"ROC AUC: {roc_auc:.4f}\n"
    )

    f.write(
        f"Sensitivity: {sensitivity:.4f}\n"
    )

    f.write(
        f"Specificity: {specificity:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"NPV: {npv:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n\n"
    )


    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"True Negatives: {tn}\n"
    )

    f.write(
        f"False Positives: {fp}\n"
    )

    f.write(
        f"False Negatives: {fn}\n"
    )

    f.write(
        f"True Positives: {tp}\n\n"
    )


    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        classification_results
    )


# ==========================================================
# 24. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 65)
print("LOOCV COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"\nAccuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC AUC:           {roc_auc:.4f}")
print(f"Sensitivity:       {sensitivity:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"F1 Score:          {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(
    f"\nAll results saved in:\n{output_dir}"
)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(output_dir)
):
    print(f"  - {filename}")