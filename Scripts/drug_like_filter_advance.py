import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ------------------------------
# Step 1: Detect input Excel file
# ------------------------------
xlsx_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.xlsx')]

if not xlsx_files:
    raise FileNotFoundError("No .xlsx file found in the current directory.")
elif len(xlsx_files) > 1:
    print(f"Multiple .xlsx files found. Using the first one: {xlsx_files[0]}")

input_excel = xlsx_files[0]
print(f"Using input file: {input_excel}")

# ------------------------------
# Step 2: Load SMILES
# ------------------------------
df = pd.read_excel(input_excel)

if 'SMILES' not in df.columns:
    raise KeyError("The Excel file must have a column named 'SMILES'.")

# ------------------------------
# Step 3: Define comprehensive filter functions
# ------------------------------
def check_pains(mol):
    """PAINS filter for pan-assay interference compounds"""
    if mol is None:
        return None
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        return int(not catalog.HasMatch(mol))
    except:
        return None

def check_brenk_filters(mol):
    """Brenk filters for problematic substructures"""
    if mol is None:
        return None
    
    unwanted_smarts = [
        # Reactive functional groups
        '[CX3](=O)[Cl,Br,I]',    # Acid halides
        '[SX2](=O)(=O)[Cl]',     # Sulfonyl chlorides
        '[NX3][Cl]',             # Chloroamines
        # Toxicity alerts
        'c1ccccc1N=O',           # Nitroaromatics
        'C#N',                   # Nitriles
        'S(=O)(=O)C',            # Sulfones
        # Metabolism issues
        'c1ccccc1OC',            # Aryl methyl ethers
        'C(=O)OC',               # Methyl esters
        'N#C',                   # Isonitriles
        '[SH]',                  # Thiols
    ]
    
    for smarts in unwanted_smarts:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return 0
        except:
            continue
    return 1

def check_lilly_rules(mol):
    """Lilly MedChem rules for unwanted substructures"""
    if mol is None:
        return None
    
    lilly_smarts = [
        '[*;!H][F,Cl,Br,I]',     # Alkyl halides (except CF3)
        '[S;!$(S(=O)=O)]',       # Thiols and sulfides
        'c1ccc2c(c1)ccc3ccccc32', # Polycyclic aromatics
        '[NH0]=C(N)N',           # Guanidines
        'C1=CC=CC=C1C2=CC=CC=C2', # Biphenyls
    ]
    
    violations = 0
    for smarts in lilly_smarts:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                violations += 1
        except:
            continue
    
    return violations

def check_reos_rules(mol):
    """REOS rules from Vertex Pharmaceuticals"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        num_rings = Lipinski.RingCount(mol)
        
        rules = [
            mw >= 200 and mw <= 500,           # Molecular weight
            logp >= -2 and logp <= 5,          # LogP
            h_donors <= 5,                     # H-bond donors
            h_acceptors <= 10,                 # H-bond acceptors
            rot_bonds <= 8,                    # Rotatable bonds
            abs(formal_charge) <= 2,           # Formal charge
            num_rings <= 6,                    # Ring count
        ]
        
        return sum(rules)  # Number of passed rules
    except:
        return None

def check_gsk_rule(mol):
    """GSK's rule: MW <= 400 and LogP <= 4"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        return int(mw <= 400 and logp <= 4)
    except:
        return None

def check_golden_triangle(mol):
    """Golden Triangle for oral bioavailability"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = MolSurf.TPSA(mol)
        
        # Check if in the "Golden Triangle"
        in_triangle = (mw >= 200 and mw <= 500 and 
                      logp >= 0 and logp <= 5 and 
                      tpsa >= 40 and tpsa <= 130)
        
        return int(in_triangle)
    except:
        return None

def check_astex_rule(mol):
    """Astex Rule of 3 for lead compounds"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        h_donors = Lipinski.NumHDonors(mol)
        tpsa = MolSurf.TPSA(mol)
        
        return int(mw <= 300 and logp <= 3 and rot_bonds <= 3 and 
                  h_donors <= 3 and tpsa <= 60)
    except:
        return None

def calculate_esol(mol):
    """ESOL solubility prediction"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        ap = MolSurf.TPSA(mol)
        
        # ESOL equation
        logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * (ap/100)
        solubility = 10**logS  # mol/L
        
        return solubility
    except:
        return None

def calculate_qed(mol):
    """Quantitative Estimate of Drug-likeness"""
    if mol is None:
        return None
    
    try:
        properties = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            MolSurf.TPSA(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumHeteroatoms(mol)
        ]
        
        # Property ranges for normalization
        ranges = [
            (180, 500),    # MW
            (-2, 5),       # LogP
            (0, 5),        # HDonors
            (2, 10),       # HAcceptors
            (20, 130),     # TPSA
            (0, 8),        # RotBonds
            (0, 3),        # AromaticRings
            (1, 8)         # Heteroatoms
        ]
        
        normalized = []
        for p, (min_val, max_val) in zip(properties, ranges):
            if p < min_val:
                norm_val = 0.0
            elif p > max_val:
                norm_val = 1.0
            else:
                norm_val = (p - min_val) / (max_val - min_val)
            normalized.append(max(0.001, norm_val))
        
        # Geometric mean as QED score
        qed = np.exp(np.sum(np.log(normalized)) / len(normalized))
        return min(1.0, max(0.0, qed))
    except:
        return None

def calculate_sa_score(mol):
    """Simplified Synthetic Accessibility Score"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        ring_count = Lipinski.RingCount(mol)
        hetero_count = Lipinski.NumHeteroatoms(mol)
        stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # Simplified SA score (1-10 scale, lower is better)
        sa_score = (mw/500 + rot_bonds/10 + ring_count/5 + 
                   hetero_count/8 + stereo_centers/2)
        return min(10.0, max(1.0, sa_score))
    except:
        return None

def calculate_composite_drug_score(mol):
    """Composite drug-likeness score (0-1 scale)"""
    if mol is None:
        return None
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = MolSurf.TPSA(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        qed = calculate_qed(mol) or 0.5
        
        # Individual property scores
        mw_score = max(0, 1 - abs(mw - 350) / 300)
        logp_score = max(0, 1 - abs(logp - 2) / 3)
        tpsa_score = max(0, 1 - abs(tpsa - 80) / 100)
        rot_score = max(0, 1 - rot_bonds / 15)
        hbond_score = max(0, 1 - (abs(h_donors - 2) + abs(h_acceptors - 5)) / 10)
        
        # Weighted composite score
        composite = (mw_score * 0.15 + logp_score * 0.15 + tpsa_score * 0.15 + 
                    rot_score * 0.10 + hbond_score * 0.15 + qed * 0.30)
        
        return min(1.0, max(0.0, composite))
    except:
        return None

def get_drug_likeness_level(score):
    """Categorize drug-likeness level"""
    if score is None:
        return "Unknown"
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Poor"

def calculate_comprehensive_drug_likeness(smiles):
    """Main function to calculate all drug-likeness parameters"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([None] * 25)
    
    # Basic properties
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    tpsa = MolSurf.TPSA(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    fraction_csp3 = Lipinski.FractionCSP3(mol)
    num_rings = Lipinski.RingCount(mol)
    num_heteroatoms = Lipinski.NumHeteroatoms(mol)
    
    # Advanced filters and scores
    pains_free = check_pains(mol)
    brenk_free = check_brenk_filters(mol)
    lilly_violations = check_lilly_rules(mol)
    reos_score = check_reos_rules(mol)
    gsk_rule = check_gsk_rule(mol)
    golden_triangle = check_golden_triangle(mol)
    astex_rule = check_astex_rule(mol)
    esol_solubility = calculate_esol(mol)
    qed_score = calculate_qed(mol)
    sa_score = calculate_sa_score(mol)
    drug_score = calculate_composite_drug_score(mol)
    
    # Overall assessment
    drug_level = get_drug_likeness_level(drug_score)
    
    # Check if passes basic filters
    passes_basic = int(
        pains_free == 1 and 
        brenk_free == 1 and 
        lilly_violations == 0 and
        mw <= 500 and 
        logp <= 5 and 
        h_donors <= 5 and 
        h_acceptors <= 10
    )
    
    return pd.Series([
        mw, logp, h_donors, h_acceptors, rot_bonds, tpsa, 
        heavy_atoms, aromatic_rings, formal_charge, fraction_csp3,
        num_rings, num_heteroatoms, pains_free, brenk_free, 
        lilly_violations, reos_score, gsk_rule, golden_triangle,
        astex_rule, esol_solubility, qed_score, sa_score, 
        drug_score, drug_level, passes_basic
    ])

# ------------------------------
# Step 4: Calculate all descriptors
# ------------------------------
descriptor_columns = [
    'MolecularWeight', 'LogP', 'HDonors', 'HAcceptors', 'RotatableBonds',
    'TPSA', 'HeavyAtoms', 'AromaticRings', 'FormalCharge', 'FractionCSP3',
    'NumRings', 'NumHeteroatoms', 'PAINS_Free', 'Brenk_Free', 
    'Lilly_Violations', 'REOS_Score', 'GSK_Rule', 'Golden_Triangle',
    'Astex_Rule', 'ESOL_Solubility', 'QED_Score', 'SA_Score', 
    'Drug_Score', 'Drug_Level', 'Passes_Basic'
]

print("Calculating comprehensive drug-likeness parameters...")
df[descriptor_columns] = df['SMILES'].apply(calculate_comprehensive_drug_likeness)

# ------------------------------
# Step 5: Generate summary statistics
# ------------------------------
def generate_summary(df):
    """Generate comprehensive summary statistics"""
    total_compounds = len(df)
    valid_compounds = df[df['MolecularWeight'].notna()].shape[0]
    
    summary = {
        'Total Compounds': total_compounds,
        'Valid SMILES': valid_compounds,
        'PAINS Free (%)': f"{df['PAINS_Free'].mean() * 100:.1f}%" if valid_compounds > 0 else 'N/A',
        'Brenk Free (%)': f"{df['Brenk_Free'].mean() * 100:.1f}%" if valid_compounds > 0 else 'N/A',
        'Pass Basic Filters (%)': f"{df['Passes_Basic'].mean() * 100:.1f}%" if valid_compounds > 0 else 'N/A',
        'Average Drug Score': f"{df['Drug_Score'].mean():.3f}" if valid_compounds > 0 else 'N/A',
        'Average QED Score': f"{df['QED_Score'].mean():.3f}" if valid_compounds > 0 else 'N/A',
        'Average SA Score': f"{df['SA_Score'].mean():.2f}" if valid_compounds > 0 else 'N/A',
        'Excellent Drug Likeness': (df['Drug_Level'] == 'Excellent').sum() if valid_compounds > 0 else 0,
        'Good Drug Likeness': (df['Drug_Level'] == 'Good').sum() if valid_compounds > 0 else 0,
        'Moderate Drug Likeness': (df['Drug_Level'] == 'Moderate').sum() if valid_compounds > 0 else 0,
        'Poor Drug Likeness': (df['Drug_Level'] == 'Poor').sum() if valid_compounds > 0 else 0,
    }
    return summary

summary_stats = generate_summary(df)

# ------------------------------
# Step 6: Save output Excel with multiple sheets
# ------------------------------
output_file = os.path.join(os.getcwd(), 'comprehensive_drug_likeness_analysis.xlsx')

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Detailed analysis sheet
    df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
    
    # Summary statistics sheet
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    # Failed compounds sheet
    failed_mask = (df['Passes_Basic'] == 0) & df['MolecularWeight'].notna()
    failed_df = df[failed_mask].copy()
    if not failed_df.empty:
        failed_df['Failure_Reasons'] = failed_df.apply(
            lambda x: ', '.join(filter(None, [
                'PAINS' if x['PAINS_Free'] == 0 else '',
                'Brenk' if x['Brenk_Free'] == 0 else '',
                'Lilly' if x['Lilly_Violations'] > 0 else '',
                'MW>500' if x['MolecularWeight'] > 500 else '',
                'LogP>5' if x['LogP'] > 5 else '',
                'HDonors>5' if x['HDonors'] > 5 else '',
                'HAcceptors>10' if x['HAcceptors'] > 10 else ''
            ])), axis=1
        )
        failed_df.to_excel(writer, sheet_name='Failed Compounds', index=False)
    
    # Top candidates sheet
    top_candidates = df[df['Drug_Score'].notna()].nlargest(20, 'Drug_Score')
    top_candidates.to_excel(writer, sheet_name='Top Candidates', index=False)

# ------------------------------
# Step 7: Print results
# ------------------------------
print(f"\nComprehensive drug-likeness analysis completed!")
print(f"Results saved to: {output_file}")
print(f"\nAnalysis includes {len(descriptor_columns)} drug-likeness parameters")
print("Output file contains four sheets:")
print("1. Detailed Analysis - All calculated parameters")
print("2. Summary Statistics - Overall statistics")
print("3. Failed Compounds - Compounds failing basic filters")
print("4. Top Candidates - Top 20 compounds by drug score")

print(f"\nSummary Statistics:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")

print(f"\nValid compounds analyzed: {summary_stats['Valid SMILES']}")
print(f"Compounds passing basic filters: {summary_stats['Pass Basic Filters (%)']}")