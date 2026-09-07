import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# ---------- Step 1: Auto-detect Excel file ----------
cwd = os.getcwd()
xlsx_files = [f for f in os.listdir(cwd) if f.endswith('.xlsx')]

if not xlsx_files:
    raise FileNotFoundError("No .xlsx file found in current directory.")
elif len(xlsx_files) > 1:
    print(f"Multiple .xlsx files found. Using: {xlsx_files[0]}")
input_file = xlsx_files[0]
print(f"📂 Using input file: {input_file}")

# ---------- Step 2: Read Excel file ----------
df = pd.read_excel(input_file)

# Ensure required columns exist
required_cols = ["bioactivity_class", "Molecule ChEMBL ID", "SMILES"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"The file must contain a '{col}' column.")

# Save these columns separately (to reattach later)
id_smiles_cols = df[["Molecule ChEMBL ID", "SMILES"]]

# Separate target from descriptors
y = df["bioactivity_class"]
X = df.drop(columns=required_cols)

# Keep only numeric columns for feature selection
X_numeric = X.select_dtypes(include=["number"])

# ---------- Step 3: Low Variance Filter ----------
print("⚙ Applying low variance filter (threshold = 0.2)...")
var_selector = VarianceThreshold(threshold=0.2)
X_var_filtered = var_selector.fit_transform(X_numeric)
selected_columns = X_numeric.columns[var_selector.get_support()]
X_filtered = pd.DataFrame(X_var_filtered, columns=selected_columns)

# ---------- Step 4: Correlation Filter ----------
print("⚙ Applying correlation filter (threshold = 0.8)...")
corr_matrix = X_filtered.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [
    column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)
]
X_final = X_filtered.drop(columns=to_drop)

print(f"📉 Removed {len(to_drop)} highly correlated features.")

# ---------- Step 5: Combine all & Save ----------
final_df = pd.concat([id_smiles_cols, y, X_final], axis=1)
output_file = "filtered_descriptors.xlsx"
final_df.to_excel(output_file, index=False)
print(f"✅ Filtering complete. Output saved as '{output_file}'")
