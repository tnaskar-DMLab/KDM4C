import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
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
output_dir = os.path.join(os.getcwd(), f"knn_loo_cv_results_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 2. Load dataset (assuming you have a single file with all data)
data_file = "normalized_descriptors.xlsx"  # Update with your actual file name
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
print(f"Number of samples: {len(X)}")

# ------------------------------
# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 6. Define KNN model with OPTIMIZED hyperparameters
knn_model = KNeighborsClassifier(
    n_neighbors=8,      # Optimized from your hyperparameter tuning
    weights='uniform',  # Optimized from tuning
    p=1,                # Optimized from tuning (Manhattan distance)
    metric='minkowski',
    n_jobs=-1
)

# ------------------------------
# 7. Set up Leave-One-Out Cross-Validation
loo = LeaveOneOut()
n_samples = len(X)

print(f"\n=== Starting Leave-One-Out Cross-Validation ===")
print(f"Using optimized hyperparameters: n_neighbors=8, p=1 (Manhattan), weights='uniform'")
print(f"Total iterations: {n_samples}")

# ------------------------------
# 8. Perform LOO-CV
y_true = []
y_pred = []
y_proba = []
train_accuracies = []

for i, (train_index, test_index) in enumerate(loo.split(X_scaled)):
    if (i + 1) % 100 == 0:  # Print progress every 100 samples
        print(f"Processing sample {i + 1}/{n_samples}")
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train model
    model = KNeighborsClassifier(
        n_neighbors=8,
        weights='uniform',
        p=1,
        metric='minkowski',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Store results
    y_true.append(y_test[0])
    y_pred.append(pred[0])
    y_proba.append(proba[0])
    
    # Calculate training accuracy for this fold
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_acc)

# Convert to arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_proba = np.array(y_proba)

# ------------------------------
# 9. Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_proba)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# ------------------------------
# 10. Print comprehensive results
print("\n" + "="*60)
print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
print("="*60)
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Inactive', 'Active']))

# ------------------------------
# 11. Train final model on full dataset
print("\nTraining final model on full dataset...")
final_model = KNeighborsClassifier(
    n_neighbors=8,
    weights='uniform',
    p=1,
    metric='minkowski',
    n_jobs=-1
)
final_model.fit(X_scaled, y)

# ------------------------------
# 12. Save model and scaler
joblib.dump(final_model, os.path.join(output_dir, 'knn_final_model_optimized.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'standard_scaler.pkl'))
print("✅ Model and scaler saved")

# ------------------------------
# 13. Save all predictions and results
# Save LOO-CV predictions
predictions_df = pd.DataFrame({
    'actual': y_true,
    'predicted': y_pred,
    'probability_active': y_proba,
    'correct': (y_true == y_pred).astype(int)
})
predictions_df.to_csv(os.path.join(output_dir, 'loo_cv_predictions.csv'), index=False)

# Save metrics summary
metrics_summary = {
    'Accuracy': accuracy,
    'Balanced_Accuracy': balanced_acc,
    'ROC_AUC': roc_auc,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'Precision': precision,
    'Number_of_Samples': n_samples
}
metrics_df = pd.DataFrame([metrics_summary])
metrics_df.to_csv(os.path.join(output_dir, 'loo_cv_metrics_summary.csv'), index=False)

# ------------------------------
# 14. Create comprehensive visualizations
plt.figure(figsize=(15, 12))

# Plot 1: Confusion Matrix
plt.subplot(2, 2, 1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inactive', 'Active'])
disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
plt.title('Confusion Matrix (LOO-CV)')

# Plot 2: ROC Curve
plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.7)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (LOO-CV)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Probability Distribution
plt.subplot(2, 2, 3)
for class_label in [0, 1]:
    mask = y_true == class_label
    plt.hist(y_proba[mask], bins=30, alpha=0.7, 
             label=f'Actual Class {class_label} ({["Inactive", "Active"][class_label]})',
             density=True)
plt.xlabel('Predicted Probability of Being Active')
plt.ylabel('Density')
plt.title('Predicted Probability Distribution by Actual Class')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Training Accuracy Distribution
plt.subplot(2, 2, 4)
plt.hist(train_accuracies, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Training Accuracy per LOO Iteration')
plt.ylabel('Frequency')
plt.title('Distribution of Training Accuracies\n(Each iteration uses n-1 samples)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loo_cv_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------
# 15. Save detailed summary report
with open(os.path.join(output_dir, 'detailed_summary_report.txt'), 'w') as f:
    f.write("KNN LEAVE-ONE-OUT CROSS-VALIDATION SUMMARY REPORT\n")
    f.write("="*65 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Dataset: {data_file}\n")
    f.write(f"Total samples: {n_samples}\n")
    f.write(f"Features used: {len(features)}\n")
    f.write(f"Class distribution: {dict(zip(['Inactive', 'Active'], np.bincount(y)))}\n\n")
    
    f.write("OPTIMIZED HYPERPARAMETERS:\n")
    f.write(f"  n_neighbors: 8\n")
    f.write(f"  p: 1 (Manhattan distance)\n")
    f.write(f"  weights: uniform\n\n")
    
    f.write("LOO-CV PERFORMANCE METRICS:\n")
    f.write(f"  Accuracy: {accuracy:.4f}\n")
    f.write(f"  Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"  ROC AUC: {roc_auc:.4f}\n")
    f.write(f"  Sensitivity: {sensitivity:.4f}\n")
    f.write(f"  Specificity: {specificity:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True Negatives: {tn}\n")
    f.write(f"  False Positives: {fp}\n")
    f.write(f"  False Negatives: {fn}\n")
    f.write(f"  True Positives: {tp}\n\n")
    
    f.write("CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_true, y_pred, target_names=['Inactive', 'Active']))
    
    f.write("\nMODEL CHARACTERISTICS:\n")
    f.write(f"  Leave-One-Out provides almost unbiased performance estimate\n")
    f.write(f"  Low variance but computationally expensive\n")
    f.write(f"  Excellent for small to medium-sized datasets\n")

print(f"\n✅ All results saved in directory: {output_dir}")
print("📊 Files created:")
for file in os.listdir(output_dir):
    print(f"  - {file}")

print(f"\n🎉 Leave-One-Out cross-validation completed successfully!")
print(f"📈 Final model performance with optimized parameters:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   ROC AUC: {roc_auc:.4f}")