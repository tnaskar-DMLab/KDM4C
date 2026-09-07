import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
from openpyxl import load_workbook

# ------------------------------
# 1. Locate the Excel file
xlsx_files = [file for file in os.listdir(os.getcwd()) if file.endswith('.xlsx')]
valid_xlsx = None
for file in xlsx_files:
    if file.lower() == "normalized_descriptors.xlsx":
        try:
            load_workbook(file)  # validate Excel file
            valid_xlsx = file
            break
        except Exception as e:
            print(f"Error loading {file}: {e}")

if not valid_xlsx:
    raise FileNotFoundError("File 'normalized_descriptors.xlsx' not found in the current directory.")
print(f"Using input file: {valid_xlsx}")

# ------------------------------
# 2. Load data
df = pd.read_excel(valid_xlsx)

# ------------------------------
# 3. Define feature columns (NEW DESCRIPTORS)
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

# ------------------------------
# 4. Check target column
if "bioactivity_class" not in df.columns:
    raise KeyError("Target column 'bioactivity_class' not found in dataset.")

# ------------------------------
# 5. Prepare data
X = df[features]
y = df["bioactivity_class"].map(lambda x: 1 if str(x).lower() == "active" else 0)

# Handle missing values
X = X.fillna(X.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 7. Train KNN model
model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    p=2
)
model.fit(X_train, y_train)

# ------------------------------
# 8. Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------
# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# ------------------------------
# 10. Save confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Inactive', 'Active'],
            yticklabels=['Inactive', 'Active'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

textstr = f"""
Sensitivity: {sensitivity:.2f}
Specificity: {specificity:.2f}
Accuracy: {accuracy:.2f}
Balanced Acc: {balanced_acc:.2f}
"""
plt.gcf().text(1.05, 0.5, textstr, fontsize=12, va='center')
plt.savefig('confusion_matrix_knn.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------
# 11. Save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (KNN)')
plt.legend(loc='lower right')
plt.savefig('roc_auc_curve_knn.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------
# 12. Save model and scaler
joblib.dump(model, 'knn_classifier.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')

# ------------------------------
# 13. Save predictions
output_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability_Active": y_proba
})
output_df.to_excel('knn_predictions.xlsx', index=False)

print("\nProcessing complete. Files saved:")
print(" - knn_classifier.pkl")
print(" - standard_scaler.pkl")
print(" - confusion_matrix_knn.png")
print(" - roc_auc_curve_knn.png")
print(" - knn_predictions.xlsx")
