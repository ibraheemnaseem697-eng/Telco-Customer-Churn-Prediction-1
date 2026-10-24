import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Dataset Load (Direct URL se takay path ka koi issue na aye)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Preprocessing & Feature Engineering
# customerID drop karna
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# TotalCharges numeric conversion (empty strings handled without row dropping to keep 7043 rows)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Replace values as per Assignment specification
replace_dict = {"No internet service": "No", "No phone service": "No"}
df = df.replace(replace_dict)

# Services list for num_services feature
services_cols = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
df['num_services'] = (df[services_cols] == 'Yes').sum(axis=1)

# Target variable encoding
y = df['Churn'].map({'Yes': 1, 'No': 0})
X_raw = df.drop(columns=['Churn'])

# One-hot encoding with drop_first=True
X = pd.get_dummies(X_raw, drop_first=True, dtype=int)

# Exact 24 columns in sequence matching the assignment requirement
exact_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'num_services',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
    'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]
X = X.reindex(columns=exact_columns, fill_value=0)

# 3. Train-Test Split (80/20 with random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predictions & Evaluation
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 7. Model Artifacts Save Karna (model folder ke andar)
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(list(X.columns), 'model/columns.joblib')

# 8. Exact Output Display
print(f"X shape: {X.shape}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("Saved: model/model.joblib, model/scaler.joblib, model/columns.joblib")
print(f"Feature columns: {list(X.columns)}")