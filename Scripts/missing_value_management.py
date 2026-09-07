import os
import pandas as pd

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

# ---------- Step 3: Fill missing numeric values with column mean ----------
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

# ---------- Step 4: Save output ----------
output_file = "filled_missing_values.xlsx"
df.to_excel(output_file, index=False)
print(f"✅ Missing values filled with column means. Output saved as '{output_file}'")
