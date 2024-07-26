
import json
import numpy as np
import xgboost as xgb

# Load XGBOOST model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_cv.json")

# Conduct Bias assessment for each sensitive feature
sensitive_features = ['job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 
'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'education_secondary', 'education_tertiary', 'education_unknown', 'marital_married',
'marital_single']

def calculate_disparate_impact(y_true, y_pred):
    non_sensitive_indices = ~y_true.astype(bool)
    positive_proportion_sensitive = y_pred.mean()
    positive_proportion_non_sensitive = y_pred[non_sensitive_indices].mean()
    if positive_proportion_sensitive == 0:
        return float('nan')
    disparate_impact = positive_proportion_non_sensitive / positive_proportion_sensitive
    return disparate_impact

for sensitive_feature in sensitive_features:
    y_true_sensitive = X_test[sensitive_feature].astype(float)
    y_pred_sensitive = y_test_pred
    disparate_impact = calculate_disparate_impact(y_true_sensitive, y_pred_sensitive)
    print(f"Disparate Impact Ratio for {sensitive_feature}: {disparate_impact}")
