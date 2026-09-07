**# Machine Learning-Guided QSAR Modelling of KDM4C Inhibitors

This repository contains the datasets, scripts, trained models

"Machine Learning-Guided QSAR Modelling of 8-HQ-Based KDM4C
Inhibitors: A Novel Predictive Framework"

## Authors

Timir Naskar  
Debasish Mandal

Department of Chemistry and Biochemistry  
Thapar Institute of Engineering and Technology  
Patiala, Punjab, India

## Overview

The study integrates:

- QSAR modelling
- Machine-learning classification
- Molecular descriptor calculation
- Feature selection
- Hyperparameter optimisation
- Applicability-domain analysis
- SHAP interpretation
- Virtual screening
- Drug-likeness filtering
- Molecular docking
- Molecular dynamics simulation
- MM/GBSA calculations
- Density functional theory calculations

## Dataset

The KDM4C dataset was obtained from ChEMBL target CHEMBL6175.

Final curated dataset:

- Total compounds: 850
- Active: 482
- Inactive: 368
- Activity threshold: IC50 <= 1000 nM

## Final Machine-Learning Model

Algorithm: k-Nearest Neighbours

Final parameters:

- Number of neighbours: 8
- Weighting: uniform
- Distance metric: Manhattan
- Test accuracy: 0.83
- Balanced accuracy: 0.83
- ROC-AUC: 0.89

## Applicability Domain

Three criteria were used:

1. Descriptor-space distance
2. Squared Mahalanobis distance
3. Prediction-confidence filtering

## Virtual Screening

Initial 8-HQ library: 9,589 compounds

Sequential screening:

9589
→ 5203 within applicability domain
→ 2422 ML-predicted active compounds
→ medicinal-chemistry filtering
→ 18 prioritized compounds

## Data Availability

The datasets and scripts necessary to reproduce the machine-learning
analysis are provided in this repository.

## Citation

Citation information will be added following publication.**
