# Disease Prediction Pipeline (Heart Disease)

## Overview
This repository contains a generalized, automated machine learning pipeline configured to predict the presence of heart disease using clinical patient data (UCI Machine Learning Repository).

## Technical Architecture
The system utilizes `scikit-learn` Pipelines and ColumnTransformers to build an airtight, production-ready classification workflow:
1. **Automated Preprocessing:** Dynamically detects numeric and categorical features, applying median imputation and standard scaling respectively.
2. **Leakage Elimination:** Preprocessing logic is strictly bundled inside the modeling pipeline to guarantee no testing data leaks into the training phase.
3. **Target Standardization:** Converts multi-class disease staging into a strict binary classification problem (Healthy vs. Heart Disease) for higher predictive accuracy.
4. **Algorithm:** Trained using a balanced Random Forest Classifier.

## Performance
The baseline model achieves strong clinical predictive power:
* **ROC-AUC Score:** 0.92
* **Class 1 Recall:** 0.91 (Successfully identifies 91% of positive disease cases, minimizing critical false negatives).

## Files
* `disease_model.py`: The universal pipeline script.
* `heart_disease_uci_pipeline.pkl`: The fully bundled `.pkl` artifact containing the imputer, scaler, and model for single-step inference.
* `heart_disease_uci.csv`: The clinical dataset.
