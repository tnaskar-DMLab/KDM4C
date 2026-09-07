import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import chi2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

# =====================================
# 0️⃣ Setup
# =====================================

output_dir = os.getcwd()
print("Saving results to:", output_dir)

model = joblib.load("knn_final_model_optimized.pkl")
scaler = joblib.load("standard_scaler.pkl")

train_df = pd.read_csv("train_set.csv")
test_df = pd.read_csv("test_set.csv")

descriptor_cols = [
"BCUT2D_MWHI","PEOE_VSA13","SlogP_VSA7","PEOE_VSA5",
"fr_Ar_NH","n6HRing","ATSC2i","MIC1",
"AATSC7dv","SlogP_VSA4","ECIndex","nAcid",
"PEOE_VSA4","VSA_EState9"
]

X_train = train_df[descriptor_cols].values
X_test = test_df[descriptor_cols].values

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = model.n_neighbors
n, p = X_train_scaled.shape

# =====================================
# 1️⃣ Distance-Based AD (KNN Consistent)
# =====================================

dist_matrix = pairwise_distances(X_train_scaled)
sorted_dist = np.sort(dist_matrix, axis=1)
avg_k_dist_train = np.mean(sorted_dist[:, 1:k+1], axis=1)

distance_threshold = np.percentile(avg_k_dist_train, 95)

test_distances = pairwise_distances(X_test_scaled, X_train_scaled)
sorted_test_dist = np.sort(test_distances, axis=1)
avg_k_dist_test = np.mean(sorted_test_dist[:, :k], axis=1)

test_df["Distance_AD"] = avg_k_dist_test <= distance_threshold
test_df["avg_kNN_distance"] = avg_k_dist_test

# =====================================
# 2️⃣ Mahalanobis AD
# =====================================

mean_vec = np.mean(X_train_scaled, axis=0)
cov_matrix = np.cov(X_train_scaled, rowvar=False)
cov_inv = np.linalg.inv(cov_matrix)

def mahalanobis_distance(x):
    diff = x - mean_vec
    return np.dot(np.dot(diff, cov_inv), diff.T)

mahal_test = np.array([mahalanobis_distance(x) for x in X_test_scaled])
chi2_threshold = chi2.ppf(0.95, df=p)

test_df["Mahalanobis_AD"] = mahal_test <= chi2_threshold
test_df["Mahalanobis_distance"] = mahal_test

# =====================================
# 3️⃣ Probability Confidence Filter
# =====================================

probs = model.predict_proba(X_test_scaled)
max_prob = np.max(probs, axis=1)

prob_threshold = 0.7
test_df["Max_Probability"] = max_prob
test_df["Probability_Filter"] = max_prob >= prob_threshold

# =====================================
# 4️⃣ Final AD Decision (Combined)
# =====================================

test_df["Final_AD"] = (
    test_df["Distance_AD"] &
    test_df["Mahalanobis_AD"] &
    test_df["Probability_Filter"]
)

# Save CSV
test_df.to_csv("Full_AD_Results.csv", index=False)

# Excel split
test_df[test_df["Final_AD"]].to_excel("Molecules_Inside_AD.xlsx", index=False)
test_df[~test_df["Final_AD"]].to_excel("Molecules_Outside_AD.xlsx", index=False)

# =====================================
# 5️⃣ PCA Visualization
# =====================================

pca = PCA(n_components=2)
X_combined = np.vstack([X_train_scaled, X_test_scaled])
X_pca = pca.fit_transform(X_combined)

X_train_pca = X_pca[:len(X_train)]
X_test_pca = X_pca[len(X_train):]

plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], alpha=0.3, label="Training")
plt.scatter(X_test_pca[test_df["Final_AD"],0],
            X_test_pca[test_df["Final_AD"],1],
            marker='^', label="Inside AD")
plt.scatter(X_test_pca[~test_df["Final_AD"],0],
            X_test_pca[~test_df["Final_AD"],1],
            marker='x', label="Outside AD")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("Applicability Domain - PCA Space")
plt.tight_layout()
plt.savefig("PCA_AD_plot.png", dpi=300)
plt.close()

# =====================================
# 6️⃣ Training Distance Histogram
# =====================================

plt.figure(figsize=(8,5))
plt.hist(avg_k_dist_train, bins=30)
plt.axvline(distance_threshold, linestyle='--')
plt.xlabel("Average k-NN Distance")
plt.ylabel("Frequency")
plt.title("Distance-Based AD Threshold")
plt.tight_layout()
plt.savefig("Distance_AD_histogram.png", dpi=300)
plt.close()

# =====================================
# 7️⃣ Automatic PDF Report
# =====================================

doc = SimpleDocTemplate("Applicability_Domain_Report.pdf")
elements = []

styles = getSampleStyleSheet()
elements.append(Paragraph("<b>Applicability Domain Report</b>", styles['Title']))
elements.append(Spacer(1, 0.3*inch))

summary_text = f"""
Training molecules: {n}<br/>
Descriptors: {p}<br/>
KNN neighbors (k): {k}<br/>
Distance AD threshold (95th percentile): {distance_threshold:.4f}<br/>
Mahalanobis Chi-square threshold (95%): {chi2_threshold:.4f}<br/>
Probability threshold: {prob_threshold}<br/>
Molecules inside final AD: {test_df['Final_AD'].sum()}<br/>
Molecules outside final AD: {(~test_df['Final_AD']).sum()}
"""

elements.append(Paragraph(summary_text, styles['Normal']))
elements.append(Spacer(1, 0.5*inch))

elements.append(Image("Distance_AD_histogram.png", width=4*inch, height=3*inch))
elements.append(Spacer(1, 0.3*inch))
elements.append(Image("PCA_AD_plot.png", width=4*inch, height=3*inch))

doc.build(elements)

print("All outputs saved successfully.")