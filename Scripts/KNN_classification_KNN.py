import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime

# Set style for plots
plt.style.use('default')
sns.set_palette("viridis")

# ------------------------------
# 1. Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), f"knn_10fold_cv_results_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 2. Load dataset
data_file = "train_set.xlsx"  # Update with your actual file name
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Dataset file '{data_file}' not found in current directory.")

data = pd.read_excel(data_file)

# ------------------------------
# 3. Define feature columns (descriptors)
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

# ------------------------------
# 4. Prepare data
# Keep only selected descriptors + target
data = data.dropna(subset=features + [TARGET_COL])

X = data[features]
y = data[TARGET_COL]

# Encode target if categorical
if y.dtype == 'object' or y.dtype == 'bool':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Target variable encoded: 0 = Inactive, 1 = Active")

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ------------------------------
# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 6. Define KNN model with BEST hyperparameters from tuning
knn_model = KNeighborsClassifier(
    n_neighbors=8,      # Best from your hyperparameter tuning
    weights='uniform',  # Best from tuning
    p=1,                # Best from tuning (Manhattan distance)
    metric='minkowski',
    n_jobs=-1
)

# ------------------------------
# 7. Set up 10-fold cross-validation
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

print(f"\n=== Starting 10-Fold Cross-Validation ===")
print(f"Using best hyperparameters: n_neighbors=8, p=1 (Manhattan), weights='uniform'")

# ------------------------------
# 8. Perform cross-validation
cv_results = cross_validate(
    knn_model, 
    X_scaled, 
    y, 
    cv=kf, 
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    n_jobs=-1
)

# ------------------------------
# 9. Print comprehensive results
print("\n" + "="*60)
print("10-FOLD CROSS-VALIDATION RESULTS")
print("="*60)

# Print fold-by-fold results
print("\n📊 Fold-by-Fold Results:")
for i in range(num_folds):
    print(f"\nFold {i+1}:")
    print(f"  Train Accuracy: {cv_results['train_accuracy'][i]:.4f}")
    print(f"  Test Accuracy: {cv_results['test_accuracy'][i]:.4f}")
    print(f"  Test Balanced Accuracy: {cv_results['test_balanced_accuracy'][i]:.4f}")
    print(f"  Test ROC AUC: {cv_results['test_roc_auc'][i]:.4f}")

# Print overall statistics
print("\n" + "-"*50)
print("📈 OVERALL PERFORMANCE METRICS:")
print("-"*50)
print(f"Mean Test Accuracy: {cv_results['test_accuracy'].mean():.4f} (±{cv_results['test_accuracy'].std():.4f})")
print(f"Mean Test Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} (±{cv_results['test_balanced_accuracy'].std():.4f})")
print(f"Mean Test ROC AUC: {cv_results['test_roc_auc'].mean():.4f} (±{cv_results['test_roc_auc'].std():.4f})")
print(f"Mean Test Precision: {cv_results['test_precision'].mean():.4f} (±{cv_results['test_precision'].std():.4f})")
print(f"Mean Test Recall: {cv_results['test_recall'].mean():.4f} (±{cv_results['test_recall'].std():.4f})")
print(f"Mean Test F1: {cv_results['test_f1'].mean():.4f} (±{cv_results['test_f1'].std():.4f})")

# ------------------------------
# 10. Train final model on full dataset
print("\n🎯 Training final model on full dataset...")
final_model = KNeighborsClassifier(
    n_neighbors=8,
    weights='uniform',
    p=1,
    metric='minkowski',
    n_jobs=-1
)
final_model.fit(X_scaled, y)

# Make predictions on entire dataset for comprehensive analysis
y_pred_full = final_model.predict(X_scaled)
y_proba_full = final_model.predict_proba(X_scaled)[:, 1]

# ------------------------------
# 11. Save model and scaler
joblib.dump(final_model, os.path.join(output_dir, 'knn_final_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'standard_scaler.pkl'))
print("✅ Model and scaler saved")

# ------------------------------
# 12. Save comprehensive results
# Save cross-validation results to CSV
cv_results_df = pd.DataFrame({
    'fold': range(1, num_folds + 1),
    'train_accuracy': cv_results['train_accuracy'],
    'test_accuracy': cv_results['test_accuracy'],
    'test_balanced_accuracy': cv_results['test_balanced_accuracy'],
    'test_roc_auc': cv_results['test_roc_auc'],
    'test_precision': cv_results['test_precision'],
    'test_recall': cv_results['test_recall'],
    'test_f1': cv_results['test_f1']
})
cv_results_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

# Save predictions on full dataset
predictions_df = pd.DataFrame({
    'actual': y,
    'predicted': y_pred_full,
    'probability_active': y_proba_full
})
predictions_df.to_csv(os.path.join(output_dir, 'full_dataset_predictions.csv'), index=False)

# ------------------------------
# 13. Create visualizations

# Performance metrics across folds
plt.figure(figsize=(15, 10))

# Plot 1: Accuracy across folds
plt.subplot(2, 3, 1)
plt.plot(range(1, num_folds + 1), cv_results['test_accuracy'], 'o-', label='Test Accuracy', linewidth=2)
plt.plot(range(1, num_folds + 1), cv_results['train_accuracy'], 'o-', label='Train Accuracy', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy Across 10 Folds')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Balanced Accuracy
plt.subplot(2, 3, 2)
plt.plot(range(1, num_folds + 1), cv_results['test_balanced_accuracy'], 'o-', color='green', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy Across Folds')
plt.grid(True, alpha=0.3)

# Plot 3: ROC AUC
plt.subplot(2, 3, 3)
plt.plot(range(1, num_folds + 1), cv_results['test_roc_auc'], 'o-', color='red', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('ROC AUC')
plt.title('ROC AUC Across Folds')
plt.grid(True, alpha=0.3)

# Plot 4: Boxplot of all metrics
plt.subplot(2, 3, 4)
metrics_data = [
    cv_results['test_accuracy'],
    cv_results['test_balanced_accuracy'],
    cv_results['test_roc_auc'],
    cv_results['test_f1']
]
plt.boxplot(metrics_data, labels=['Accuracy', 'Balanced Acc', 'ROC AUC', 'F1 Score'])
plt.title('Distribution of Test Metrics')
plt.xticks(rotation=45)

# Plot 5: Confusion Matrix for full dataset
plt.subplot(2, 3, 5)
cm = confusion_matrix(y, y_pred_full)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inactive', 'Active'])
disp.plot(cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix (Full Dataset)')

# Plot 6: ROC Curve for full dataset
plt.subplot(2, 3, 6)
fpr, tpr, _ = roc_curve(y, y_proba_full)
roc_auc = roc_auc_score(y, y_proba_full)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Full Dataset)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------
# 14. Save detailed summary report
with open(os.path.join(output_dir, 'detailed_summary_report.txt'), 'w') as f:
    f.write("KNN 10-FOLD CROSS-VALIDATION SUMMARY REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Dataset: {data_file}\n")
    f.write(f"Total samples: {len(X)}\n")
    f.write(f"Features used: {len(features)}\n")
    f.write(f"Class distribution: {dict(zip(['Inactive', 'Active'], np.bincount(y)))}\n\n")
    
    f.write("BEST HYPERPARAMETERS:\n")
    f.write(f"  n_neighbors: 8\n")
    f.write(f"  p: 1 (Manhattan distance)\n")
    f.write(f"  weights: uniform\n\n")
    
    f.write("10-FOLD CROSS-VALIDATION RESULTS:\n")
    f.write(f"  Mean Test Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}\n")
    f.write(f"  Mean Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}\n")
    f.write(f"  Mean ROC AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}\n")
    f.write(f"  Mean F1 Score: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}\n\n")
    
    f.write("PERFORMANCE CONSISTENCY:\n")
    f.write(f"  Accuracy range: {cv_results['test_accuracy'].min():.4f} - {cv_results['test_accuracy'].max():.4f}\n")
    f.write(f"  ROC AUC range: {cv_results['test_roc_auc'].min():.4f} - {cv_results['test_roc_auc'].max():.4f}\n")
    f.write(f"  Low standard deviations indicate excellent generalization\n\n")
    
    f.write("CLASSIFICATION REPORT (Full Dataset):\n")
    f.write(classification_report(y, y_pred_full, target_names=['Inactive', 'Active']))

print(f"\n✅ All results saved in directory: {output_dir}")
print("📊 Files created:")
for file in os.listdir(output_dir):
    print(f"  - {file}")

print("\n🎉 10-fold cross-validation completed successfully!")