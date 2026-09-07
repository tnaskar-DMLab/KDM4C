import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

# ---------- Step 1: Auto-detect .xlsx file in cwd ----------
cwd = os.getcwd()
xlsx_files = [f for f in os.listdir(cwd) if f.endswith('.xlsx')]

if not xlsx_files:
    raise FileNotFoundError("No .xlsx file found in current directory.")
elif len(xlsx_files) > 1:
    print(f"Multiple .xlsx files found. Using: {xlsx_files[0]}")
input_file = xlsx_files[0]
print(f"📂 Using input file: {input_file}")

# ---------- Step 2: Read file ----------
df = pd.read_excel(input_file)

if 'SMILES' not in df.columns:
    raise ValueError("Input file must contain a 'SMILES' column.")

# Convert SMILES to RDKit molecules
mols = [Chem.MolFromSmiles(s) for s in df['SMILES']]

# ---------- Step 3: RDKit descriptors ----------
def calc_rdkit_descriptors(mols):
    rdkit_desc_names = [d[0] for d in Descriptors.descList]
    rdkit_data = []
    for mol in mols:
        if mol is None:
            rdkit_data.append([None] * len(rdkit_desc_names))
        else:
            rdkit_data.append([desc_func(mol) for _, desc_func in Descriptors.descList])
    return pd.DataFrame(rdkit_data, columns=rdkit_desc_names)

print("⚙ Calculating RDKit descriptors...")
rdkit_df = calc_rdkit_descriptors(mols)

# ---------- Step 4: Mordred descriptors ----------
print("⚙ Calculating Mordred descriptors (ignore_3D=True)...")
calc = Calculator(descriptors, ignore_3D=True)
mordred_df = calc.pandas(mols)

# Convert Mordred's object types to strings/numbers
mordred_df = mordred_df.applymap(lambda x: float(x) if x is not None else None)

# ---------- Step 5: Merge descriptors ----------
final_df = pd.concat([df, rdkit_df, mordred_df], axis=1)

# ---------- Step 6: Save output ----------
output_file = "descriptors_rdkit_mordred.csv"
final_df.to_csv(output_file, index=False)
print(f"✅ Descriptor calculation completed. Saved as '{output_file}'")
