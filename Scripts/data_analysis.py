import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================================
# Create Output Folder
# ==========================================================

output_dir = "chemical_space_analysis"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Detect Excel File
# ==========================================================

xlsx_files = [f for f in os.listdir() if f.endswith(".xlsx")]

if not xlsx_files:
    raise FileNotFoundError("No Excel file found in current directory.")

input_file = xlsx_files[0]
print(f"Using file: {input_file}")

# ==========================================================
# Read Data
# ==========================================================

df = pd.read_excel(input_file)

print(f"\nDataset Shape: {df.shape}")

# Remove duplicate columns if present
df = df.loc[:, ~df.columns.duplicated()]

print(f"Shape after duplicate removal: {df.shape}")

# ==========================================================
# Check Required Columns
# ==========================================================

if "SMILES" not in df.columns:
    raise ValueError("SMILES column not found.")

if "bioactivity" not in df.columns:
    raise ValueError("bioactivity column not found.")

# ==========================================================
# Activity Classification
# ==========================================================

def classify_bioactivity(value):
    try:
        value = float(value)

        if value <= 1000:
            return "active"
        else:
            return "inactive"

    except:
        return "undefined"

df["bioactivity_class"] = df["bioactivity"].apply(classify_bioactivity)

# ==========================================================
# Lipinski Descriptor Calculation
# ==========================================================

required_desc = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors"
]

missing_desc = [c for c in required_desc if c not in df.columns]

if len(missing_desc) > 0:

    print("\nCalculating missing Lipinski descriptors...")

    descriptor_data = []

    for smi in df["SMILES"]:

        mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            descriptor_data.append(
                [np.nan, np.nan, np.nan, np.nan]
            )
            continue

        descriptor_data.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol)
        ])

    lipinski_df = pd.DataFrame(
        descriptor_data,
        columns=required_desc
    )

    for col in required_desc:

        if col not in df.columns:
            df[col] = lipinski_df[col]

else:
    print("\nUsing existing Lipinski descriptors.")

# ==========================================================
# Remove Duplicates Again
# ==========================================================

df = df.loc[:, ~df.columns.duplicated()]

# ==========================================================
# Descriptor Columns
# ==========================================================

desc_cols = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors"
]

# Convert to numeric

for col in desc_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================================
# Lipinski Violations
# ==========================================================

df["Lipinski_Violations"] = (
    (df["MW"] > 500).astype(int)
    + (df["LogP"] > 5).astype(int)
    + (df["NumHDonors"] > 5).astype(int)
    + (df["NumHAcceptors"] > 10).astype(int)
)

df["Lipinski_Pass"] = np.where(
    df["Lipinski_Violations"] <= 1,
    "Pass",
    "Fail"
)

# ==========================================================
# Split Dataset
# ==========================================================

active_df = df[
    df["bioactivity_class"] == "active"
]

inactive_df = df[
    df["bioactivity_class"] == "inactive"
]

# ==========================================================
# Save Excel Files
# ==========================================================

df.to_excel(
    "classified_with_lipinski.xlsx",
    index=False
)

active_df.to_excel(
    "active_compounds.xlsx",
    index=False
)

inactive_df.to_excel(
    "inactive_compounds.xlsx",
    index=False
)

# ==========================================================
# Summary Statistics
# ==========================================================

with pd.ExcelWriter(
    os.path.join(output_dir,
                 "summary_statistics.xlsx")
) as writer:

    df[desc_cols].describe().to_excel(
        writer,
        sheet_name="All"
    )

    active_df[desc_cols].describe().to_excel(
        writer,
        sheet_name="Active"
    )

    inactive_df[desc_cols].describe().to_excel(
        writer,
        sheet_name="Inactive"
    )

# ==========================================================
# Class Distribution
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    x="bioactivity_class",
    data=df
)

plt.title("Class Distribution")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir,
                 "class_distribution_bar.png"),
    dpi=300
)

plt.close()

# ==========================================================
# Pie Chart
# ==========================================================

counts = df["bioactivity_class"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%"
)

plt.title("Class Distribution")

plt.savefig(
    os.path.join(output_dir,
                 "class_distribution_pie.png"),
    dpi=300
)

plt.close()

# ==========================================================
# Histograms
# ==========================================================

for col in desc_cols:

    plt.figure(figsize=(7,5))

    sns.histplot(
        df[col].dropna(),
        bins=30,
        kde=True
    )

    plt.title(f"{col} Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{col}_histogram.png"
        ),
        dpi=300
    )

    plt.close()

# ==========================================================
# Boxplots
# ==========================================================

for col in desc_cols:

    plt.figure(figsize=(6,5))

    sns.boxplot(
        x="bioactivity_class",
        y=col,
        data=df
    )

    plt.title(f"{col}: Active vs Inactive")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{col}_boxplot.png"
        ),
        dpi=300
    )

    plt.close()

# ==========================================================
# MW vs LogP
# ==========================================================

plt.figure(figsize=(7,6))

sns.scatterplot(
    data=df,
    x="MW",
    y="LogP",
    hue="bioactivity_class"
)

plt.title("Chemical Space: MW vs LogP")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir,
                 "MW_vs_LogP.png"),
    dpi=300
)

plt.close()

# ==========================================================
# HBA vs HBD
# ==========================================================

plt.figure(figsize=(7,6))

sns.scatterplot(
    data=df,
    x="NumHAcceptors",
    y="NumHDonors",
    hue="bioactivity_class"
)

plt.title("HBA vs HBD")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir,
                 "HBA_vs_HBD.png"),
    dpi=300
)

plt.close()

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df[desc_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir,
                 "correlation_heatmap.png"),
    dpi=300
)

plt.close()

# ==========================================================
# PCA Analysis
# ==========================================================

pca_df = df[desc_cols].dropna()

if len(pca_df) > 5:

    scaler = StandardScaler()

    scaled = scaler.fit_transform(pca_df)

    pca = PCA(n_components=2)

    pcs = pca.fit_transform(scaled)

    pca_result = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"]
    )

    pca_result["Activity"] = df.loc[
        pca_df.index,
        "bioactivity_class"
    ].values

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=pca_result,
        x="PC1",
        y="PC2",
        hue="Activity"
    )

    plt.title(
        f"PCA Chemical Space\n"
        f"PC1={pca.explained_variance_ratio_[0]*100:.1f}% "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir,
                     "PCA_Chemical_Space.png"),
        dpi=300
    )

    plt.close()

# ==========================================================
# Lipinski Violation Plot
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    x="Lipinski_Violations",
    data=df
)

plt.title("Lipinski Violations")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir,
                 "Lipinski_Violations.png"),
    dpi=300
)

plt.close()

# ==========================================================
# Pass/Fail Summary
# ==========================================================

df["Lipinski_Pass"].value_counts().to_excel(
    os.path.join(
        output_dir,
        "Lipinski_Pass_Fail.xlsx"
    )
)

# ==========================================================
# Final Summary
# ==========================================================

print("\n==============================")
print("ANALYSIS COMPLETED")
print("==============================")

print(f"Total compounds : {len(df)}")
print(f"Active          : {len(active_df)}")
print(f"Inactive        : {len(inactive_df)}")

print("\nLipinski Summary")
print(df["Lipinski_Pass"].value_counts())

print(f"\nResults saved in:")
print(os.path.abspath(output_dir))