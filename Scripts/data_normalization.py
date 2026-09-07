import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------- Step 1: Auto-detect Excel file ----------
cwd = os.getcwd()
xlsx_files = [f for f in os.listdir(cwd) if f.endswith('.xlsx')]

if not xlsx_files:
    raise FileNotFoundError("No .xlsx file found in current directory.")
elif len(xlsx_files) > 1:
    print(f"Multiple .xlsx files found. Using: {xlsx_files[0]}")
input_excel = xlsx_files[0]
print(f"📂 Using Excel input file: {input_excel}")

# ---------- Step 2: Read Excel file ----------
df_excel = pd.read_excel(input_excel)

# Ensure required columns exist
required_cols = ["bioactivity_class", "Molecule ChEMBL ID", "SMILES"]
for col in required_cols:
    if col not in df_excel.columns:
        raise ValueError(f"The Excel file must contain a '{col}' column.")

# Keep ID & target columns
id_smiles_cols = df_excel[["Molecule ChEMBL ID", "SMILES"]]
y = df_excel["bioactivity_class"]

# Select only numeric descriptor columns
X = df_excel.drop(columns=required_cols)
X_numeric = X.select_dtypes(include=["number"])

# ---------- Step 3: Apply Min-Max Scaling on Excel ----------
print("⚙ Applying Min-Max scaling (0–1) on Excel file...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)

# Combine & save normalized Excel
final_excel_df = pd.concat([id_smiles_cols, y, X_scaled_df], axis=1)
output_excel_file = "normalized_descriptors.xlsx"
final_excel_df.to_excel(output_excel_file, index=False)
print(f"✅ Normalized Excel saved as '{output_excel_file}'")

# ---------- Step 4: Normalize filtered_descriptors.csv ----------
csv_files = [f for f in os.listdir(cwd) if f.endswith('.csv') and "filtered_descriptors" in f]
if not csv_files:
    raise FileNotFoundError("No filtered_descriptors CSV found in cwd.")
filtered_csv_file = csv_files[0]
print(f"📂 Using filtered CSV file: {filtered_csv_file}")

# Read filtered CSV
df_filtered = pd.read_csv(filtered_csv_file)

# Keep non-numeric columns (like SMILES) unchanged
non_numeric_cols = df_filtered.select_dtypes(exclude=["float64", "int64"]).columns

# Detect common numeric descriptors between Excel and filtered CSV
common_numeric_cols = [col for col in df_filtered.columns if col in X_numeric.columns]

# Apply min-max scaling using the same scaler (based on Excel)
print("⚙ Applying Min-Max scaling on filtered CSV based on Excel reference...")
df_filtered_scaled = df_filtered.copy()
for col in common_numeric_cols:
    min_val = X_numeric[col].min()
    max_val = X_numeric[col].max()
    if max_val - min_val == 0:
        df_filtered_scaled[col] = 0.0
    else:
        df_filtered_scaled[col] = (df_filtered[col] - min_val) / (max_val - min_val)

# Save scaled filtered descriptors
output_csv_scaled = "scaled_filtered_descriptors.xlsx"
df_filtered_scaled.to_excel(output_csv_scaled, index=False)
print(f"✅ Scaled filtered descriptors saved as '{output_csv_scaled}'")
