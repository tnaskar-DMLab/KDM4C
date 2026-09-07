import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (make_scorer, balanced_accuracy_score, accuracy_score, 
                           roc_auc_score, classification_report, confusion_matrix, 
                           roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
import joblib
from datetime import datetime

# Set style for plots
plt.style.use('default')
sns.set_palette("viridis")

# ------------------------------
# 1. Input files
train_file = "train_set.xlsx"
test_file = "test_set.xlsx"

if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise FileNotFoundError("Train or test Excel file not found in cwd.")

# ------------------------------
# 2. Load datasets
train_df = pd.read_excel(train_file)
test_df = pd.read_excel(test_file)

# ------------------------------
# 3. Feature columns
features = [
    "BCUT2D_MWHI", "PEOE_VSA13", "SlogP_VSA7", "PEOE_VSA5", "fr_Ar_NH",
    "n6HRing", "ATSC2i", "MIC1", "AATSC7dv", "SlogP_VSA4",
    "ECIndex", "nAcid", "PEOE_VSA4", "VSA_EState9"
]

X_train = train_df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = train_df["bioactivity_class"].map(lambda x: 1 if str(x).lower() == "active" or x == 1 else 0)

X_test = test_df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = test_df["bioactivity_class"].map(lambda x: 1 if str(x).lower() == "active" or x == 1 else 0)

# ------------------------------
# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5. Define KNN and hyperparameter grid
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': list(range(3, 16)),  # test k=3 to 15
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan or Euclidean
}

# ------------------------------
# 6. Balanced accuracy as scoring
scorer = make_scorer(balanced_accuracy_score)

# ------------------------------
# 7. GridSearchCV
print("Starting hyperparameter tuning...")
grid = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train)

# ------------------------------
# 8. Best parameters
print("Best Hyperparameters found:")
print(grid.best_params_)
print(f"Best Cross-Validated Balanced Accuracy: {grid.best_score_:.4f}")

# ------------------------------
# 9. Evaluate on test set
best_knn = grid.best_estimator_
y_pred_test = best_knn.predict(X_test_scaled)
y_proba_test = best_knn.predict_proba(X_test_scaled)[:, 1]

acc_test = accuracy_score(y_test, y_pred_test)
ba_test = balanced_accuracy_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_proba_test)
cm_test = confusion_matrix(y_test, y_pred_test)

print(f"\nTest Set Accuracy: {acc_test:.4f}")
print(f"Test Set Balanced Accuracy: {ba_test:.4f}")
print(f"Test Set ROC AUC: {roc_auc_test:.4f}")
print("Test Set Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", cm_test)

# ------------------------------
# 10. Create directory for outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"knn_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 11. Save predictions and probabilities
results_df = test_df.copy()
results_df['predicted_class'] = y_pred_test
results_df['predicted_probability'] = y_proba_test
results_df['actual_class'] = y_test

# Save to CSV
results_df.to_csv(os.path.join(output_dir, 'predictions_and_probabilities.csv'), index=False)

# ------------------------------
# 12. Create and save plots

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_display = ConfusionMatrixDisplay.from_estimator(best_knn, X_test_scaled, y_test, 
                                                  display_labels=['Inactive', 'Active'],
                                                  cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_test:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
plt.plot(recall, precision, label='Precision-Recall Curve', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Learning Curve
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    best_knn, X_train_scaled, y_train, cv=5, 
    scoring='balanced_accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

plt.xlabel('Training Set Size')
plt.ylabel('Balanced Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Validation Curve for n_neighbors (using best other parameters)
plt.figure(figsize=(10, 6))
param_range = np.arange(3, 21)
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(weights=grid.best_params_['weights'], p=grid.best_params_['p']),
    X_train_scaled, y_train, 
    param_name="n_neighbors", param_range=param_range,
    cv=5, scoring="balanced_accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, label="Training score", color="r")
plt.plot(param_range, test_mean, label="Cross-validation score", color="g")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

plt.axvline(x=grid.best_params_['n_neighbors'], color='blue', linestyle='--', 
           label=f'Best k: {grid.best_params_["n_neighbors"]}')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Balanced Accuracy")
plt.title("Validation Curve for k (Number of Neighbors)")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'validation_curve_k.png'), dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------
# 13. Save model, scaler, and results summary
joblib.dump(best_knn, os.path.join(output_dir, 'knn_best_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler_for_knn.pkl'))
joblib.dump(grid, os.path.join(output_dir, 'grid_search_results.pkl'))

# Save results summary
with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
    f.write("KNN CLASSIFICATION RESULTS SUMMARY\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write("Best Hyperparameters:\n")
    for param, value in grid.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nBest CV Balanced Accuracy: {grid.best_score_:.4f}\n")
    f.write(f"Test Set Accuracy: {acc_test:.4f}\n")
    f.write(f"Test Set Balanced Accuracy: {ba_test:.4f}\n")
    f.write(f"Test Set ROC AUC: {roc_auc_test:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_test))
    f.write(f"\nConfusion Matrix:\n{cm_test}")

# ------------------------------
# 14. Save grid search results as CSV
cv_results_df = pd.DataFrame(grid.cv_results_)
cv_results_df.to_csv(os.path.join(output_dir, 'grid_search_cv_results.csv'), index=False)

print(f"\nAll results, plots, and models saved in directory: {output_dir}")
print("Files created:")
for file in os.listdir(output_dir):
    print(f"  - {file}")