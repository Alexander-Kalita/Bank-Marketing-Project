
import shap
import json
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Load XGBOOST model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_cv.json")

# Provide SHAP values prediction explanations
customer = {
    'features': {
        'age': 18,
        'balance': 300,
        'day': 5,
        'duration': 500,
        'campaign': 1,
        'pdays': -1,
        'previous': 4,
        'job_blue-collar': 0,
        'job_entrepreneur': 0,
        'job_housemaid': 0,
        'job_management': 1,
        'job_retired': 0,
        'job_self-employed': 0,
        'job_services': 0,
        'job_student': 0,
        'job_technician': 0,
        'job_unemployed': 0,
        'job_unknown': 0,
        'education_secondary': 1,
        'education_tertiary': 0,
        'education_unknown': 0,
        'marital_married': 0,
        'marital_single': 1,
        'default_yes': 1,
        'housing_yes': 1,
        'loan_yes': 1,
        'contact_telephone': 1,
        'contact_unknown': 0,
        'month_aug': 0,
        'month_dec': 0,
        'month_feb': 0,
        'month_jan': 0,
        'month_jul': 1,
        'month_jun': 0,
        'month_mar': 0,
        'month_may': 0,
        'month_nov': 0,
        'month_oct': 0,
        'month_sep': 0,
        'poutcome_other': 0,
        'poutcome_success': 1,
        'poutcome_unknown': 0
    }
}
features_dict = customer['features']
features_df = pd.DataFrame([features_dict])
explainer = shap.Explainer(loaded_model)
shap_values = explainer.shap_values(features_df)
feature_names = features_df.columns
for i, sample_shap_values in enumerate(shap_values):
    print(f"SHAP values:")
    for feature_name, shap_value in zip(feature_names, sample_shap_values):
        print(f"{feature_name}: {shap_value}")

# Create SHAP values visualization plot
shap_values_instance = shap_values[0]
abs_shap_values = np.abs(shap_values_instance)
sorted_indices = np.argsort(abs_shap_values)
top_features_indices = sorted_indices[-10:][::-1]
top_shap_values = shap_values_instance[top_features_indices]
top_feature_names = [feature_names[i] for i in top_features_indices]
plt.figure(figsize=(10, 6))
bars = plt.barh(top_feature_names, top_shap_values, color="skyblue")
for bar, shap_value in zip(bars, top_shap_values):
    if shap_value < 0:
        plt.text(bar.get_x() + bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2, f'{shap_value:.2f}',
                 ha='center', va='center', fontsize=8)
    else:
        plt.text(bar.get_x() + bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f'{shap_value:.2f}',
                 ha='center', va='center', fontsize=8)
plt.xlabel("SHAP Value")
plt.yticks(fontsize=9)
plt.title("Figure 1:Top 10 SHAP Values for Prediction Instance", fontsize=14)
plt.gca().invert_yaxis()
plt.show()
