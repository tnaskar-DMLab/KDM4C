import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score


# ==========================================================
# SETTINGS
# ==========================================================

DATA_FILE = "train_set.xlsx"

TARGET_COL = "bioactivity_class"

FEATURES = [
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

# KNN parameters from optimized model
N_NEIGHBORS = 8
P_VALUE = 1
WEIGHTS = "uniform"

# Number of Y-randomization repetitions
N_PERMUTATIONS = 1000

# Reproducibility
RANDOM_SEED = 42


# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = os.path.join(
    os.getcwd(),
    f"knn_y_randomization_{timestamp}"
)

os.makedirs(output_dir, exist_ok=True)

print(f"\nOutput directory:\n{output_dir}")


# ==========================================================
# LOAD DATA
# ==========================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found in current directory."
    )

data = pd.read_excel(DATA_FILE)

required_columns = FEATURES + [TARGET_COL]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# Remove missing values
data = data.dropna(
    subset=required_columns
).reset_index(drop=True)


# ==========================================================
# PREPARE X AND y
# ==========================================================

X = data[FEATURES].to_numpy(dtype=float)

y_raw = data[TARGET_COL]


# ----------------------------------------------------------
# Encode activity labels
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

    else:

        raise ValueError(
            "Unexpected target labels found: "
            f"{sorted(unique_labels)}"
        )

else:

    y = y_raw.to_numpy(dtype=int)


if not set(np.unique(y)).issubset({0, 1}):
    raise ValueError(
        f"Target must contain only 0/1. Found: {np.unique(y)}"
    )


# ==========================================================
# DATA INFORMATION
# ==========================================================

class_counts = np.bincount(y, minlength=2)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Number of compounds: {len(y)}")
print(f"Number of descriptors: {X.shape[1]}")
print(f"Inactive compounds: {class_counts[0]}")
print(f"Active compounds:   {class_counts[1]}")

if len(y) != 680:
    print(
        f"\nWARNING: Expected 680 training compounds, "
        f"but found {len(y)}."
    )


# ==========================================================
# MODEL PIPELINE
# ==========================================================

# StandardScaler is inside Pipeline.
# Therefore scaling is fitted separately inside each CV fold.

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS,
            p=P_VALUE,
            metric="minkowski",
            n_jobs=-1
        )
    )

])


# ==========================================================
# CROSS-VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_SEED
)


# ==========================================================
# SCORING METRICS
# ==========================================================

scoring = {

    "accuracy":
        "accuracy",

    "balanced_accuracy":
        make_scorer(
            balanced_accuracy_score
        ),

    "roc_auc":
        "roc_auc",

    "f1":
        make_scorer(
            f1_score
        )
}


# ==========================================================
# ORIGINAL MODEL PERFORMANCE
# ==========================================================

print("\n" + "=" * 70)
print("ORIGINAL MODEL - 10-FOLD CROSS-VALIDATION")
print("=" * 70)

original_cv = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


original_accuracy = np.mean(
    original_cv["test_accuracy"]
)

original_balanced_accuracy = np.mean(
    original_cv["test_balanced_accuracy"]
)

original_roc_auc = np.mean(
    original_cv["test_roc_auc"]
)

original_f1 = np.mean(
    original_cv["test_f1"]
)


print(
    f"Accuracy:          "
    f"{original_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{original_balanced_accuracy:.4f}"
)

print(
    f"ROC AUC:           "
    f"{original_roc_auc:.4f}"
)

print(
    f"F1 Score:          "
    f"{original_f1:.4f}"
)


# ==========================================================
# Y-RANDOMIZATION
# ==========================================================

print("\n" + "=" * 70)
print("STARTING Y-RANDOMIZATION")
print("=" * 70)

print(
    f"Number of permutations: {N_PERMUTATIONS}"
)

rng = np.random.default_rng(
    RANDOM_SEED
)


randomized_results = []


for permutation in range(
    1,
    N_PERMUTATIONS + 1
):

    # ------------------------------------------------------
    # Shuffle ONLY y labels
    # X remains completely unchanged
    # ------------------------------------------------------

    y_random = rng.permutation(y)


    # ------------------------------------------------------
    # Same model + same CV procedure
    # ------------------------------------------------------

    cv_result = cross_validate(
        model,
        X,
        y_random,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )


    randomized_results.append({

        "Permutation":
            permutation,

        "Accuracy":
            np.mean(
                cv_result["test_accuracy"]
            ),

        "Balanced_Accuracy":
            np.mean(
                cv_result[
                    "test_balanced_accuracy"
                ]
            ),

        "ROC_AUC":
            np.mean(
                cv_result["test_roc_auc"]
            ),

        "F1":
            np.mean(
                cv_result["test_f1"]
            )
    })


    if (
        permutation % 100 == 0
        or permutation == N_PERMUTATIONS
    ):

        print(
            f"Completed "
            f"{permutation}/"
            f"{N_PERMUTATIONS}"
        )


# ==========================================================
# CONVERT TO DATAFRAME
# ==========================================================

randomized_df = pd.DataFrame(
    randomized_results
)


# ==========================================================
# RANDOMIZED MODEL SUMMARY
# ==========================================================

random_accuracy_mean = (
    randomized_df["Accuracy"].mean()
)

random_accuracy_sd = (
    randomized_df["Accuracy"].std()
)

random_balanced_mean = (
    randomized_df[
        "Balanced_Accuracy"
    ].mean()
)

random_balanced_sd = (
    randomized_df[
        "Balanced_Accuracy"
    ].std()
)

random_auc_mean = (
    randomized_df["ROC_AUC"].mean()
)

random_auc_sd = (
    randomized_df["ROC_AUC"].std()
)

random_f1_mean = (
    randomized_df["F1"].mean()
)

random_f1_sd = (
    randomized_df["F1"].std()
)


# ==========================================================
# EMPIRICAL P-VALUES
# ==========================================================

# How often did randomized models perform
# as well as or better than the real model?

p_accuracy = (
    1
    + np.sum(
        randomized_df["Accuracy"]
        >= original_accuracy
    )
) / (
    N_PERMUTATIONS + 1
)


p_balanced = (
    1
    + np.sum(
        randomized_df[
            "Balanced_Accuracy"
        ]
        >= original_balanced_accuracy
    )
) / (
    N_PERMUTATIONS + 1
)


p_auc = (
    1
    + np.sum(
        randomized_df["ROC_AUC"]
        >= original_roc_auc
    )
) / (
    N_PERMUTATIONS + 1
)


p_f1 = (
    1
    + np.sum(
        randomized_df["F1"]
        >= original_f1
    )
) / (
    N_PERMUTATIONS + 1
)


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("Y-RANDOMIZATION RESULTS")
print("=" * 70)

print("\nORIGINAL MODEL")

print(
    f"Accuracy:          "
    f"{original_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{original_balanced_accuracy:.4f}"
)

print(
    f"ROC AUC:           "
    f"{original_roc_auc:.4f}"
)

print(
    f"F1 Score:          "
    f"{original_f1:.4f}"
)


print("\nRANDOMIZED MODELS")

print(
    f"Accuracy:          "
    f"{random_accuracy_mean:.4f} "
    f"± {random_accuracy_sd:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{random_balanced_mean:.4f} "
    f"± {random_balanced_sd:.4f}"
)

print(
    f"ROC AUC:           "
    f"{random_auc_mean:.4f} "
    f"± {random_auc_sd:.4f}"
)

print(
    f"F1 Score:          "
    f"{random_f1_mean:.4f} "
    f"± {random_f1_sd:.4f}"
)


print("\nEMPIRICAL P-VALUES")

print(
    f"Accuracy:          {p_accuracy:.6f}"
)

print(
    f"Balanced Accuracy: {p_balanced:.6f}"
)

print(
    f"ROC AUC:           {p_auc:.6f}"
)

print(
    f"F1 Score:          {p_f1:.6f}"
)


# ==========================================================
# SAVE RANDOMIZED RESULTS
# ==========================================================

randomized_df.to_csv(
    os.path.join(
        output_dir,
        "y_randomization_all_permutations.csv"
    ),
    index=False
)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "ROC AUC",
        "F1 Score"
    ],

    "Original_Model": [
        original_accuracy,
        original_balanced_accuracy,
        original_roc_auc,
        original_f1
    ],

    "Randomized_Mean": [
        random_accuracy_mean,
        random_balanced_mean,
        random_auc_mean,
        random_f1_mean
    ],

    "Randomized_SD": [
        random_accuracy_sd,
        random_balanced_sd,
        random_auc_sd,
        random_f1_sd
    ],

    "Empirical_P_Value": [
        p_accuracy,
        p_balanced,
        p_auc,
        p_f1
    ]

})


summary.to_csv(
    os.path.join(
        output_dir,
        "y_randomization_summary.csv"
    ),
    index=False
)


# ==========================================================
# PLOT 1 - ROC AUC
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    randomized_df["ROC_AUC"],
    bins=30,
    edgecolor="black"
)

plt.axvline(
    original_roc_auc,
    linewidth=3,
    linestyle="--",
    label=(
        f"Original model "
        f"(AUC={original_roc_auc:.3f})"
    )
)

plt.xlabel(
    "ROC AUC"
)

plt.ylabel(
    "Number of Randomized Models"
)

plt.title(
    "KNN Y-Randomization Test"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "y_randomization_ROC_AUC.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# PLOT 2 - BALANCED ACCURACY
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    randomized_df[
        "Balanced_Accuracy"
    ],
    bins=30,
    edgecolor="black"
)

plt.axvline(
    original_balanced_accuracy,
    linewidth=3,
    linestyle="--",
    label=(
        f"Original model "
        f"(BA={original_balanced_accuracy:.3f})"
    )
)

plt.xlabel(
    "Balanced Accuracy"
)

plt.ylabel(
    "Number of Randomized Models"
)

plt.title(
    "KNN Y-Randomization Test"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "y_randomization_balanced_accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# DETAILED TEXT REPORT
# ==========================================================

report_file = os.path.join(
    output_dir,
    "y_randomization_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "KNN Y-RANDOMIZATION TEST\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Dataset: {DATA_FILE}\n"
    )

    f.write(
        f"Number of compounds: "
        f"{len(y)}\n"
    )

    f.write(
        f"Number of descriptors: "
        f"{len(FEATURES)}\n"
    )

    f.write(
        f"Y-randomization iterations: "
        f"{N_PERMUTATIONS}\n\n"
    )

    f.write(
        "MODEL PARAMETERS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"n_neighbors = {N_NEIGHBORS}\n"
    )

    f.write(
        f"p = {P_VALUE} "
        f"(Manhattan distance)\n"
    )

    f.write(
        f"weights = {WEIGHTS}\n\n"
    )


    f.write(
        "VALIDATION PROCEDURE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Y labels were randomly permuted while "
        "the descriptor matrix X remained unchanged.\n"
    )

    f.write(
        "Each randomized dataset was evaluated "
        "using stratified 10-fold cross-validation.\n"
    )

    f.write(
        "StandardScaler was fitted within each "
        "cross-validation training fold to prevent "
        "information leakage.\n"
    )

    f.write(
        "The independent external test set was "
        "not used during Y-randomization.\n\n"
    )


    f.write(
        "ORIGINAL MODEL\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Accuracy: "
        f"{original_accuracy:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{original_balanced_accuracy:.4f}\n"
    )

    f.write(
        f"ROC AUC: "
        f"{original_roc_auc:.4f}\n"
    )

    f.write(
        f"F1 Score: "
        f"{original_f1:.4f}\n\n"
    )


    f.write(
        "Y-RANDOMIZED MODELS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Accuracy: "
        f"{random_accuracy_mean:.4f} "
        f"± {random_accuracy_sd:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{random_balanced_mean:.4f} "
        f"± {random_balanced_sd:.4f}\n"
    )

    f.write(
        f"ROC AUC: "
        f"{random_auc_mean:.4f} "
        f"± {random_auc_sd:.4f}\n"
    )

    f.write(
        f"F1 Score: "
        f"{random_f1_mean:.4f} "
        f"± {random_f1_sd:.4f}\n\n"
    )


    f.write(
        "EMPIRICAL P-VALUES\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Accuracy: "
        f"{p_accuracy:.6f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{p_balanced:.6f}\n"
    )

    f.write(
        f"ROC AUC: "
        f"{p_auc:.6f}\n"
    )

    f.write(
        f"F1 Score: "
        f"{p_f1:.6f}\n"
    )


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 70)
print("Y-RANDOMIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nResults saved in:\n{output_dir}"
)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(output_dir)
):
    print(f"  - {filename}")
