import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Load SMILES from Excel and calculate descriptors
# ------------------------------
input_excel = "test_molecule.xlsx"

if not os.path.exists(input_excel):
    raise FileNotFoundError(f"{input_excel} not found in the current working directory.")

# Read SMILES column
smiles_df = pd.read_excel(input_excel)
if "SMILES" not in smiles_df.columns:
    raise ValueError("The input Excel must contain a 'SMILES' column.")

smiles_list = smiles_df["SMILES"].dropna().tolist()
if not smiles_list:
    raise ValueError("No SMILES found in test_molecule.xlsx!")

print(f"📂 Loaded {len(smiles_list)} molecules from {input_excel}")

# RDKit descriptor function
def calc_rdkit_descriptors(mol):
    rdkit_desc_names = [d[0] for d in Descriptors.descList]
    rdkit_values = [desc_func(mol) for _, desc_func in Descriptors.descList]
    return pd.DataFrame([rdkit_values], columns=rdkit_desc_names)

# Mordred calculator
calc = Calculator(descriptors, ignore_3D=True)

# Store results
all_desc = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"⚠ Skipping invalid SMILES: {smi}")
        continue

    print(f"⚙ Calculating descriptors for {smi}...")

    # RDKit descriptors
    rdkit_df = calc_rdkit_descriptors(mol)

    # Mordred descriptors
    mordred_df = calc.pandas([mol])
    mordred_df = mordred_df.applymap(lambda x: float(x) if x is not None else None)

    # Merge with SMILES
    input_df = pd.DataFrame({'SMILES': [smi]})
    full_df = pd.concat([input_df, rdkit_df, mordred_df], axis=1)

    all_desc.append(full_df)

# Combine all molecules into one DataFrame
if not all_desc:
    raise ValueError("No valid molecules to process.")
full_df = pd.concat(all_desc, ignore_index=True)

# Save full descriptor file
descriptor_file = "descriptor_test_molecule.csv"
full_df.to_csv(descriptor_file, index=False)
print(f"✅ Descriptor calculation completed. Saved as '{descriptor_file}'")

# ------------------------------
# Step 2: Filter specific columns
# ------------------------------
columns_to_keep = [
    "BCUT2D_MWHI","PEOE_VSA13","SlogP_VSA7","PEOE_VSA5","fr_Ar_NH","n6HRing",
    "ATSC2i","MIC1","AATSC7dv","SlogP_VSA4","ECIndex","nAcid","PEOE_VSA4","VSA_EState9"
]

existing_columns = [col for col in columns_to_keep if col in full_df.columns]
missing_columns = set(columns_to_keep) - set(existing_columns)
if missing_columns:
    print(f"⚠ Warning: Missing columns in descriptors: {missing_columns}")

columns_to_output = ['SMILES'] + existing_columns
filtered_df = full_df[columns_to_output]

filtered_file = "filtered_descriptors.csv"
filtered_df.to_csv(filtered_file, index=False)
print(f"✅ Filtered dataframe saved as '{filtered_file}'")

# ------------------------------
# Step 2b: Enhanced Min-max scaling based on reference
# ------------------------------
normalized_file = "normalized_descriptors.xlsx"
output_file = "scaled_filtered_descriptors.xlsx"

# Check files exist
for f in [normalized_file, filtered_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found in the current working directory.")

print("🔄 Performing min-max scaling based on reference data...")

try:
    # Load reference normalized descriptors
    if normalized_file.endswith('.csv'):
        normalized_df = pd.read_csv(normalized_file)
    else:
        normalized_df = pd.read_excel(normalized_file)
    
    # Load filtered descriptors
    filtered_df_loaded = pd.read_csv(filtered_file)
    
    # Display dataset info
    print(f"   Reference data shape: {normalized_df.shape}")
    print(f"   Test data shape: {filtered_df_loaded.shape}")
    
    # Detect numeric columns in both datasets
    normalized_numeric_cols = normalized_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    filtered_numeric_cols = filtered_df_loaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    print(f"   Numeric columns in reference: {len(normalized_numeric_cols)}")
    print(f"   Numeric columns in test: {len(filtered_numeric_cols)}")
    
    # Columns common to both (excluding SMILES if present)
    common_numeric_cols = [col for col in filtered_numeric_cols 
                          if col in normalized_numeric_cols and col != 'SMILES']
    
    if not common_numeric_cols:
        # Try to find columns with different case or spaces
        normalized_cols_lower = [col.lower().strip() for col in normalized_numeric_cols]
        filtered_cols_lower = [col.lower().strip() for col in filtered_numeric_cols]
        
        common_cols_lower = set(normalized_cols_lower) & set(filtered_cols_lower)
        
        if common_cols_lower:
            # Map back to original column names
            normalized_col_map = {col.lower().strip(): col for col in normalized_numeric_cols}
            filtered_col_map = {col.lower().strip(): col for col in filtered_numeric_cols}
            
            common_numeric_cols = [filtered_col_map[col] for col in common_cols_lower]
            print(f"   Found common columns after case normalization: {common_numeric_cols}")
    
    if not common_numeric_cols:
        raise ValueError("No common numeric columns found between reference and test data!")
    
    print(f"   Common columns for scaling: {common_numeric_cols}")
    
    # Handle missing values in both datasets
    normalized_df_clean = normalized_df[common_numeric_cols].fillna(normalized_df[common_numeric_cols].mean())
    filtered_df_clean = filtered_df_loaded[common_numeric_cols].fillna(filtered_df_loaded[common_numeric_cols].mean())
    
    # Enhanced min-max scaling function with validation
    def robust_min_max_scale(ref_df, target_df, cols):
        scaled_df = target_df.copy()
        scaling_info = {}
        
        for col in cols:
            try:
                # Get min and max from reference data
                min_val = ref_df[col].min()
                max_val = ref_df[col].max()
                
                # Handle edge cases
                if pd.isna(min_val) or pd.isna(max_val):
                    print(f"   ⚠ Column '{col}' has NaN in reference data, skipping scaling")
                    scaling_info[col] = {'min': None, 'max': None, 'scaled': False}
                    continue
                
                if max_val - min_val == 0:
                    print(f"   ⚠ Column '{col}' has zero range in reference data, setting to 0")
                    scaled_df[col] = 0.0
                    scaling_info[col] = {'min': min_val, 'max': max_val, 'scaled': True, 'zero_range': True}
                else:
                    # Apply min-max scaling
                    scaled_df[col] = (target_df[col] - min_val) / (max_val - min_val)
                    scaling_info[col] = {'min': min_val, 'max': max_val, 'scaled': True, 'zero_range': False}
                    
                    # Check for outliers in test data
                    test_min = target_df[col].min()
                    test_max = target_df[col].max()
                    if test_min < min_val or test_max > max_val:
                        print(f"   ⚠ Column '{col}' has test values outside reference range: "
                              f"[{test_min:.3f}, {test_max:.3f}] vs ref [{min_val:.3f}, {max_val:.3f}]")
                        
            except Exception as e:
                print(f"   ❌ Error scaling column '{col}': {e}")
                scaling_info[col] = {'min': None, 'max': None, 'scaled': False, 'error': str(e)}
        
        return scaled_df, scaling_info
    
    # Apply scaling
    scaled_data, scaling_info = robust_min_max_scale(normalized_df_clean, filtered_df_clean, common_numeric_cols)
    
    # Count successfully scaled columns
    scaled_count = sum(1 for info in scaling_info.values() if info.get('scaled', False))
    print(f"   Successfully scaled {scaled_count}/{len(common_numeric_cols)} columns")
    
    # Create final scaled dataframe with all original columns
    scaled_filtered_df = filtered_df_loaded.copy()
    for col in common_numeric_cols:
        if col in scaled_data.columns:
            scaled_filtered_df[col] = scaled_data[col]
    
    # Save scaling information
    scaling_df = pd.DataFrame.from_dict(scaling_info, orient='index')
    scaling_df.to_csv('scaling_information.csv', index=True)
    print("   ✅ Scaling information saved to 'scaling_information.csv'")
    
    # Save scaled output
    scaled_filtered_df.to_excel(output_file, index=False)
    print(f"   ✅ Scaled filtered descriptors saved to '{output_file}'")
    
    # Basic validation
    print("   📊 Scaled data summary:")
    for col in common_numeric_cols[:5]:  # Show first 5 columns
        if col in scaled_filtered_df.columns:
            col_min = scaled_filtered_df[col].min()
            col_max = scaled_filtered_df[col].max()
            print(f"      {col}: [{col_min:.3f}, {col_max:.3f}]")
    
except Exception as e:
    print(f"❌ Error during min-max scaling: {e}")
    print("⚠ Falling back to using unscaled data for prediction")
    
    # Use unscaled data as fallback
    scaled_filtered_df = pd.read_csv(filtered_file)
    scaled_filtered_df.to_excel(output_file, index=False)
    print("   Using original (unscaled) data for prediction")

# ------------------------------
# Step 3: Load pre-trained KNN model with OPTIMIZED hyperparameters
# ------------------------------
# Try multiple possible model file names
model_files = [
    'knn_final_model_optimized.pkl',  # From LOO-CV
    'knn_best_model.pkl',             # From hyperparameter tuning
    'knn_classifier.pkl'              # Default name
]

scaler_files = [
    'standard_scaler.pkl',
    'scaler_for_knn.pkl'
]

model = None
scaler = None

# Try to load model
for model_file in model_files:
    if os.path.exists(model_file):
        try:
            model = joblib.load(model_file)
            print(f"✅ Loaded model from: {model_file}")
            break
        except:
            continue

# Try to load scaler
for scaler_file in scaler_files:
    if os.path.exists(scaler_file):
        try:
            scaler = joblib.load(scaler_file)
            print(f"✅ Loaded scaler from: {scaler_file}")
            break
        except:
            continue

if model is None:
    raise FileNotFoundError("No pre-trained KNN model found. Please train a model first.")

if scaler is None:
    raise FileNotFoundError("No scaler file found.")

print("✅ Loaded pre-trained KNN model and scaler.")

# ------------------------------
# Step 4: Prepare features and make predictions
# ------------------------------
features = columns_to_keep

# Check if all required features are present
missing_cols = [col for col in features if col not in scaled_filtered_df.columns]
if missing_cols:
    print(f"⚠ Warning: Missing descriptor columns: {missing_cols}")
    print("Available columns:", scaled_filtered_df.columns.tolist())
    
    # Use only available features
    available_features = [col for col in features if col in scaled_filtered_df.columns]
    print(f"Using available features: {available_features}")
else:
    available_features = features

# Handle missing values
X = scaled_filtered_df[available_features].fillna(scaled_filtered_df[available_features].mean())

# Standardize features using the loaded scaler
X_scaled = scaler.transform(X)

# ------------------------------
# Step 5: Make predictions with OPTIMIZED hyperparameters
# ------------------------------
print("\n🎯 Making predictions with optimized hyperparameters...")

# Predict bioactivity
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# Add predictions to dataframe
scaled_filtered_df['Predicted_Class'] = y_pred
scaled_filtered_df['Predicted_Class_Label'] = ['Active' if p == 1 else 'Inactive' for p in y_pred]
scaled_filtered_df['Probability_Active'] = y_proba
scaled_filtered_df['Confidence_Level'] = ['High' if prob > 0.7 else 'Medium' if prob > 0.6 else 'Low' for prob in y_proba]

# ------------------------------
# Step 6: Enhanced Applicability Domain Check using kNN Distance Method
# ------------------------------
print("🔍 Performing enhanced applicability domain check using kNN distance method...")

try:
    # Load training data for AD check
    train_data = pd.read_excel('normalized_descriptors.xlsx')
    X_train = train_data[available_features].fillna(train_data[available_features].mean())
    X_train_scaled = scaler.transform(X_train)
    
    # Use the same k as your optimized KNN model
    k = model.n_neighbors if hasattr(model, 'n_neighbors') else 8
    
    # Fit kNN on training set
    nbrs = NearestNeighbors(n_neighbors=k, metric='cityblock').fit(X_train_scaled)
    
    # Compute mean neighbor distances for training set
    train_distances, _ = nbrs.kneighbors(X_train_scaled)
    train_mean_distances = train_distances.mean(axis=1)
    
    # Compute mean neighbor distances for test set
    test_distances, _ = nbrs.kneighbors(X_scaled)
    test_mean_distances = test_distances.mean(axis=1)
    
    # Define AD cutoff (mean + 3*std of training distances)
    cutoff = train_mean_distances.mean() + 3 * train_mean_distances.std()
    
    # Flag samples
    in_domain = test_mean_distances <= cutoff
    
    # Add AD results to dataframe
    scaled_filtered_df['Mean_Distance_To_Neighbors'] = test_mean_distances
    scaled_filtered_df['In_Applicability_Domain'] = in_domain
    scaled_filtered_df['AD_Cutoff_Distance'] = cutoff
    scaled_filtered_df['Domain_Confidence'] = ['High' if dist <= cutoff * 0.7 
                                            else 'Medium' if dist <= cutoff 
                                            else 'Low' for dist in test_mean_distances]
    
    print(f"   kNN distance method: k={k}, metric=cityblock (Manhattan)")
    print(f"   Applicability domain cutoff: {cutoff:.4f}")
    print(f"   Compounds within AD: {np.sum(in_domain)}/{len(in_domain)}")
    print(f"   Compounds outside AD: {np.sum(~in_domain)}/{len(in_domain)}")
    
    # Create visualizations
    def create_ad_visualizations(train_distances, test_distances, cutoff, results_df):
        """Create visualizations for applicability domain analysis"""
        viz_dir = "ad_visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.hist(train_distances, bins=30, alpha=0.7, label="Training Set", density=True)
        plt.hist(test_distances, bins=30, alpha=0.7, label="Test Compounds", density=True)
        plt.axvline(cutoff, color="red", linestyle="--", label=f"AD Cutoff = {cutoff:.2f}")
        plt.xlabel("Mean Distance to k Nearest Neighbors")
        plt.ylabel("Density")
        plt.title("Applicability Domain (kNN Distance Method)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'ad_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualizations saved to '{viz_dir}' directory")
    
    create_ad_visualizations(train_mean_distances, test_mean_distances, cutoff, scaled_filtered_df)
    
except Exception as e:
    print(f"⚠ Could not perform enhanced applicability domain check: {e}")

# ------------------------------
# Step 7: Save comprehensive results
# ------------------------------
# Save predictions
prediction_file = 'predicted_activity_optimized.csv'
scaled_filtered_df.to_csv(prediction_file, index=False)
print(f"✅ Predictions saved to '{prediction_file}'")

# Save detailed Excel report
excel_file = 'predicted_activity_detailed.xlsx'
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    scaled_filtered_df.to_excel(writer, sheet_name='All_Compounds', index=False)
    print(f"✅ Detailed Excel report saved to '{excel_file}'")

# ------------------------------
# Step 8: Print summary
# ------------------------------
print("\n" + "="*60)
print("PREDICTION SUMMARY")
print("="*60)
print(f"Total molecules processed: {len(scaled_filtered_df)}")
print(f"Predicted as Active: {np.sum(y_pred == 1)}")
print(f"Predicted as Inactive: {np.sum(y_pred == 0)}")
print(f"Average probability of activity: {np.mean(y_proba):.3f}")

if 'In_Applicability_Domain' in scaled_filtered_df.columns:
    in_domain_count = np.sum(scaled_filtered_df['In_Applicability_Domain'])
    print(f"Compounds within applicability domain: {in_domain_count}/{len(scaled_filtered_df)}")

print(f"\n✅ Prediction completed successfully!")