import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           make_scorer, roc_curve, ConfusionMatrixDisplay)
import joblib
from datetime import datetime

# Set style for plots
plt.style.use('default')
sns.set_palette("viridis")

# -----------------------------
# Load your dataset
# -----------------------------
data = pd.read_excel("train_set.xlsx")   # update file path if needed

# -----------------------------
# Define the descriptors you want to use (updated list)
# -----------------------------
selected_features = [
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

TARGET_COL = "bioactivity_class"   # change if needed

# -----------------------------
# Keep only selected descriptors + target
# -----------------------------
data = data.dropna(subset=selected_features + [TARGET_COL])

X = data[selected_features]
y = data[TARGET_COL]

# Encode target if categorical
if y.dtype == 'object' or y.dtype == 'bool':
    le = LabelEncoder()
    y = le.fit_transform(y)

# -----------------------------
# Feature scaling (important for KNN)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Define classifier with BEST HYPERPARAMETERS from tuning
# -----------------------------
knn_classifier = KNeighborsClassifier(
    n_neighbors=8,      # Best from tuning
    weights='uniform',  # Best from tuning
    p=1,                # Best from tuning (Manhattan distance)
    metric='minkowski'
)

# -----------------------------
# Create output directory
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"knn_cross_validation_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Multiple scoring metrics for cross-validation
# -----------------------------
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# -----------------------------
# Enhanced K-Fold Cross Validation with multiple metrics
# -----------------------------
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

print("Running Enhanced 5-Fold Cross Validation with Best Hyperparameters...")
print(f"Hyperparameters: n_neighbors=8, p=1 (Manhattan), weights='uniform'")

# Perform cross-validation with multiple metrics
cv_results = cross_validate(
    knn_classifier, 
    X_scaled, 
    y, 
    cv=kf, 
    scoring=scoring,
    return_train_score=True,
    return_estimator=True
)

# -----------------------------
# Print comprehensive results
# -----------------------------
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS WITH BEST HYPERPARAMETERS")
print("="*60)

# Print fold-by-fold results
print("\nFold-by-Fold Results:")
for i in range(num_folds):
    print(f"\nFold {i+1}:")
    print(f"  Train Accuracy: {cv_results['train_accuracy'][i]:.4f}")
    print(f"  Test Accuracy: {cv_results['test_accuracy'][i]:.4f}")
    print(f"  Test Balanced Accuracy: {cv_results['test_balanced_accuracy'][i]:.4f}")
    print(f"  Test ROC AUC: {cv_results['test_roc_auc'][i]:.4f}")

# Print overall statistics
print("\n" + "-"*40)
print("OVERALL PERFORMANCE METRICS:")
print("-"*40)
print(f"Mean Test Accuracy: {cv_results['test_accuracy'].mean():.4f} (±{cv_results['test_accuracy'].std():.4f})")
print(f"Mean Test Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} (±{cv_results['test_balanced_accuracy'].std():.4f})")
print(f"Mean Test ROC AUC: {cv_results['test_roc_auc'].mean():.4f} (±{cv_results['test_roc_auc'].std():.4f})")
print(f"Mean Test Precision: {cv_results['test_precision'].mean():.4f} (±{cv_results['test_precision'].std():.4f})")
print(f"Mean Test Recall: {cv_results['test_recall'].mean():.4f} (±{cv_results['test_recall'].std():.4f})")
print(f"Mean Test F1: {cv_results['test_f1'].mean():.4f} (±{cv_results['test_f1'].std():.4f})")

# -----------------------------
# Train final model on full dataset
# -----------------------------
print("\nTraining final model on full dataset...")
final_model = KNeighborsClassifier(
    n_neighbors=8,
    weights='uniform',
    p=1,
    metric='minkowski'
)
final_model.fit(X_scaled, y)

# -----------------------------
# Save model and results
# -----------------------------
joblib.dump(final_model, os.path.join(output_dir, 'knn_final_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

# Save results to CSV
results_df = pd.DataFrame({
    'fold': range(1, num_folds + 1),
    'train_accuracy': cv_results['train_accuracy'],
    'test_accuracy': cv_results['test_accuracy'],
    'test_balanced_accuracy': cv_results['test_balanced_accuracy'],
    'test_roc_auc': cv_results['test_roc_auc'],
    'test_precision': cv_results['test_precision'],
    'test_recall': cv_results['test_recall'],
    'test_f1': cv_results['test_f1']
})
results_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

# -----------------------------
# Create performance visualization
# -----------------------------
plt.figure(figsize=(12, 8))

# Plot accuracy across folds
plt.subplot(2, 2, 1)
plt.plot(range(1, num_folds + 1), cv_results['test_accuracy'], 'o-', label='Test Accuracy')
plt.plot(range(1, num_folds + 1), cv_results['train_accuracy'], 'o-', label='Train Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy Across Folds')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot balanced accuracy
plt.subplot(2, 2, 2)
plt.plot(range(1, num_folds + 1), cv_results['test_balanced_accuracy'], 'o-', color='green')
plt.xlabel('Fold')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy Across Folds')
plt.grid(True, alpha=0.3)

# Plot ROC AUC
plt.subplot(2, 2, 3)
plt.plot(range(1, num_folds + 1), cv_results['test_roc_auc'], 'o-', color='red')
plt.xlabel('Fold')
plt.ylabel('ROC AUC')
plt.title('ROC AUC Across Folds')
plt.grid(True, alpha=0.3)

# Boxplot of all test metrics
plt.subplot(2, 2, 4)
metrics_data = [
    cv_results['test_accuracy'],
    cv_results['test_balanced_accuracy'],
    cv_results['test_roc_auc'],
    cv_results['test_f1']
]
plt.boxplot(metrics_data, labels=['Accuracy', 'Balanced Acc', 'ROC AUC', 'F1 Score'])
plt.title('Distribution of Test Metrics')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cross_validation_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# Save summary report
# -----------------------------
with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
    f.write("KNN CROSS-VALIDATION SUMMARY REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Dataset: normalized_descriptors.xlsx\n")
    f.write(f"Total samples: {len(X)}\n")
    f.write(f"Features used: {len(selected_features)}\n\n")
    
    f.write("BEST HYPERPARAMETERS:\n")
    f.write(f"  n_neighbors: 8\n")
    f.write(f"  p: 1 (Manhattan distance)\n")
    f.write(f"  weights: uniform\n\n")
    
    f.write("CROSS-VALIDATION RESULTS (5-fold):\n")
    f.write(f"  Mean Test Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}\n")
    f.write(f"  Mean Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}\n")
    f.write(f"  Mean ROC AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}\n")
    f.write(f"  Mean F1 Score: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}\n\n")
    
    f.write("PERFORMANCE CONSISTENCY:\n")
    f.write(f"  Accuracy range: {cv_results['test_accuracy'].min():.4f} - {cv_results['test_accuracy'].max():.4f}\n")
    f.write(f"  Low standard deviation indicates good generalization\n")

print(f"\nAll results saved in directory: {output_dir}")
print("Cross-validation completed successfully! ✅")