import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

# ----------------------------
# Config
# ----------------------------
BASE_NAME = "normalized_descriptors"     # Excel file base name
TARGET_COLUMN = "bioactivity_class"      # Target column
ALLOWED_EXTS = [".xlsx", ".xls"]         # Allowed extensions
CV_FOLDS = 5
RANDOM_STATE = 42
MAX_FEATURES = None  # limit forward selection steps if needed

# Columns that should never be used as descriptors
METADATA_COLS = [
    TARGET_COLUMN, "bioactivity", "smiles", "SMILES", "canonical_smiles",
    "molecule_name", "compound_name", "id", "ID", "molecule_id", "MoleculeID"
]

# ----------------------------
# Locate Excel file robustly
# ----------------------------
def find_excel_file(base_name: str, exts):
    for ext in exts:
        p = os.path.join(os.getcwd(), base_name + ext)
        if os.path.isfile(p):
            return p
    patterns = [f"**/{base_name}{ext}" for ext in exts]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    if candidates:
        candidates.sort(key=lambda s: len(s))
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find an Excel file named '{base_name}' with extensions {exts}."
    )

# ----------------------------
# Load data
# ----------------------------
print(f"Searching for Excel file '{BASE_NAME}' ...")
filename = find_excel_file(BASE_NAME, ALLOWED_EXTS)
print(f"Loading data from: {filename}")
df = pd.read_excel(filename)

if TARGET_COLUMN not in df.columns:
    raise KeyError(f"Target column '{TARGET_COLUMN}' not found in file.")

# Drop rows with missing target
df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

# Extract target
y = df[TARGET_COLUMN]

# Drop metadata columns and keep only numeric descriptors
exclude_cols = [col for col in METADATA_COLS if col in df.columns]
X = df.drop(columns=exclude_cols).select_dtypes(include=[np.number])
if X.shape[1] == 0:
    raise ValueError("No numeric descriptor columns found after dropping metadata.")

print(f"Using {X.shape[1]} numeric descriptor features for modeling.")

# Handle missing feature values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# ----------------------------
# Define Gradient Boosting model
# ----------------------------
models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

# ----------------------------
# Forward Feature Selection
# ----------------------------
def forward_feature_selection(model, Xdf, y_arr, max_features=None, cv_folds=5):
    if max_features is None:
        max_features = Xdf.shape[1]
    selected_features = []
    best_scores = []
    all_features = list(Xdf.columns)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for _ in range(max_features):
        scores_with_candidates = []
        for feat in all_features:
            if feat not in selected_features:
                candidate_feats = selected_features + [feat]
                try:
                    score = cross_val_score(
                        model, Xdf[candidate_feats], y_arr, cv=cv, scoring='accuracy'
                    ).mean()
                except Exception as e:
                    print(f"Error with feature {feat}: {e}")
                    score = 0
                scores_with_candidates.append((score, feat))

        if not scores_with_candidates:
            break
        scores_with_candidates.sort(reverse=True, key=lambda t: t[0])
        best_score, best_feat = scores_with_candidates[0]
        if selected_features and best_score <= best_scores[-1]:
            break
        selected_features.append(best_feat)
        best_scores.append(best_score)
        print(f"Added feature: {best_feat}, CV Accuracy: {best_score:.4f}")

    return selected_features, best_scores

# ----------------------------
# Outputs
# ----------------------------
output_folder = os.getcwd()
plot_folder = output_folder
summary_accuracy = pd.DataFrame()
summary_features = {}

# ----------------------------
# Run Gradient Boosting forward selection
# ----------------------------
for model_name, model in models.items():
    print(f"\n=== Forward Feature Selection for {model_name} ===")
    selected_feats, accuracies = forward_feature_selection(
        model, X_scaled, y_encoded, max_features=MAX_FEATURES, cv_folds=CV_FOLDS
    )

    df_acc = pd.DataFrame({
        'Num_Features': list(range(1, len(selected_feats) + 1)),
        'Accuracy': accuracies
    })
    df_feats = pd.DataFrame({'Selected_Features': selected_feats})

    max_acc = max(accuracies) if accuracies else np.nan
    summary_accuracy.loc[model_name, 'Max_Accuracy'] = max_acc
    summary_accuracy.loc[model_name, 'Num_Selected_Features'] = len(selected_feats)
    summary_features[model_name] = selected_feats

    acc_filename = os.path.join(output_folder, f"{model_name}_accuracy.xlsx")
    feats_filename = os.path.join(output_folder, f"{model_name}_selected_features.xlsx")
    df_acc.to_excel(acc_filename, index=False)
    df_feats.to_excel(feats_filename, index=False)
    print(f"Saved: {acc_filename}")
    print(f"Saved: {feats_filename}")

    if len(df_acc) > 0:
        plt.figure()
        plt.plot(df_acc['Num_Features'], df_acc['Accuracy'], marker='o')
        plt.title(f"Accuracy vs Number of Features - {model_name}")
        plt.xlabel("Number of Features")
        plt.ylabel("Cross-Validated Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plot_folder, f"{model_name}_accuracy_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved accuracy plot to {plot_path}")
    else:
        print("No features selected; skipping plot.")

summary_accuracy_filename = os.path.join(output_folder, "summary_accuracy.xlsx")
summary_accuracy.to_excel(summary_accuracy_filename)
print(f"\nSaved summary accuracy to {summary_accuracy_filename}")

summary_feats_filename = os.path.join(output_folder, "summary_selected_features.xlsx")
with pd.ExcelWriter(summary_feats_filename) as writer:
    for model_name, features_list in summary_features.items():
        pd.DataFrame(features_list, columns=["Feature"]).to_excel(
            writer, sheet_name=model_name, index=False
        )
print(f"Saved summary selected features to {summary_feats_filename}")

print("\nAll done!")
