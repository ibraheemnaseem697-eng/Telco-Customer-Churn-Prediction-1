from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Model artifacts load karein (jo train_model.py se model/ folder me bane)
model = joblib.load('model/model.joblib')
scaler = joblib.load('model/scaler.joblib')
model_columns = joblib.load('model/columns.joblib')

SERVICES = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

def prepare_customer(customer_data, columns):
    data = pd.DataFrame([customer_data])
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)
    data["tenure"] = data["tenure"].astype(int)
    data["MonthlyCharges"] = data["MonthlyCharges"].astype(float)
    data["TotalCharges"] = data["TotalCharges"].astype(float)
    
    # Text mapping as per Assignment
    data = data.replace({"No internet service": "No", "No phone service": "No"})
    data["num_services"] = (data[SERVICES] == "Yes").sum(axis=1)
    
    # One-hot encoding & reindexing to exact 24 columns
    data = pd.get_dummies(data, dtype=int)
    data = data.reindex(columns=columns, fill_value=0)
    return data

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_title = None
    prediction_desc = None
    prediction_class = None
    error_message = None

    if request.method == 'POST':
        try:
            customer_data = {
                'gender': request.form.get('gender'),
                'SeniorCitizen': request.form.get('SeniorCitizen', 0),
                'Partner': request.form.get('Partner'),
                'Dependents': request.form.get('Dependents'),
                'tenure': request.form.get('tenure', 0),
                'PhoneService': request.form.get('PhoneService'),
                'MultipleLines': request.form.get('MultipleLines'),
                'InternetService': request.form.get('InternetService'),
                'OnlineSecurity': request.form.get('OnlineSecurity'),
                'OnlineBackup': request.form.get('OnlineBackup'),
                'DeviceProtection': request.form.get('DeviceProtection'),
                'TechSupport': request.form.get('TechSupport'),
                'StreamingTV': request.form.get('StreamingTV'),
                'StreamingMovies': request.form.get('StreamingMovies'),
                'Contract': request.form.get('Contract'),
                'PaperlessBilling': request.form.get('PaperlessBilling'),
                'PaymentMethod': request.form.get('PaymentMethod'),
                'MonthlyCharges': request.form.get('MonthlyCharges', 0),
                'TotalCharges': request.form.get('TotalCharges', 0)
            }

            # 24-feature prepared dataframe
            processed_df = prepare_customer(customer_data, model_columns)
            scaled_input = scaler.transform(processed_df)

            # Prediction & Probability
            pred = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0][1] * 100

            if pred == 1:
                prediction_title = "Prediction: Churn"
                prediction_desc = f"This customer is likely to leave the company. Chance of churn: {prob:.2f}%"
                prediction_class = "danger"
            else:
                prediction_title = "Prediction: No Churn"
                prediction_desc = f"This customer is likely to stay with the company. Chance of churn: {prob:.2f}%"
                prediction_class = "safe"

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template('index.html', 
                           title=prediction_title, 
                           desc=prediction_desc, 
                           p_class=prediction_class,
                           error=error_message)

if __name__ == '__main__':
    app.run(debug=True)