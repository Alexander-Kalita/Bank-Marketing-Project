
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import warnings
import xgboost as xgb
from xgboost import XGBClassifier
from scipy.stats import yeojohnson
from mlflow.sklearn import log_model
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Load dataset
bank_data = pd.read_csv('C:/Users/Alexander_Kalita/my_mlops_project/data/bank-full.csv', sep=",")
bank_data.head()

# View data structure
print(bank_data.info())

# Change object into category type variables
for col in bank_data.select_dtypes(include='object').columns:
    bank_data[col] = bank_data [col].astype('category')
print(bank_data.dtypes)

# Separate the features and the target variable
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Transform numeric variables using the Yeo-Johnson transformation
num_cols = bank_data.select_dtypes(include=['int64']).columns
transformed_data = pd.DataFrame()
for col in num_cols:
    transformed_data[col] = yeojohnson(bank_data[col])[0]
bank_data.drop(columns=num_cols, inplace=True)
display(transformed_data.head())

# Transform categorical variables into binary variables
cat_vars = ['job', 'education', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
binary_data = pd.get_dummies(bank_data[cat_vars], columns=cat_vars, drop_first=True)
bank_data.drop(columns=cat_vars, inplace=True)
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})
X = pd.concat([bank_data, binary_data], axis=1)
y = bank_data['y']
display(X.head())

# Combine the transformed numerical and categorical variables into a single dataset
final_data = pd.concat([transformed_data, binary_data, bank_data['y']], axis=1)
X = final_data.drop(columns=['y'])
y = final_data['y']
final_data.info()

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Training data shape:', X_train.shape, y_train.shape)
print('Testing data shape:', X_test.shape, y_test.shape)

# Create a DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost hyperparameters
params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 5,
    'colsample_bytree': 0.8
}

# Build the XGBoost model with 5-fold cross-validation
xgb_cv = XGBClassifier(**params)
xgb_cv.fit(X_train, y_train)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_cv, X, y, cv=kf)
print("Cross-validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Evaluate the model on the test set
threshold = 0.2
y_test_pred_prob = xgb_cv.predict_proba(X_test)[:,1]
y_test_pred = np.where(y_test_pred_prob >= threshold, 1, 0)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Threshold:', threshold)
print('Test Accuracy:', test_accuracy)

# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost.data')

# Save and load the trained XGBoost model
xgb_cv.save_model("xgb_cv.json")
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_cv.json")
