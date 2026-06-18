# PREDICT.PY
# CUSTOMER CHURN PLATFORM
# STEP 7



# 1 . IMPORT LIBRARIES

import pandas as pd

import joblib


# 2 . Load Trained Model

model = joblib.load(r"C:\Users\aakas\OneDrive\DATA SCINCE PROJECTS\customer_churn_project\models\customer_churn_model.joblib")

print("Model Loaded Successfully")


# 3 . New Customer Data

customer_data = pd.DataFrame( {
    "gender" : ["Female"],

    "SeniorCitizen" : [0],

    "Partner" : ["Yes"],

    "Dependents": ["No"],

    "tenure": [12],

    "PhoneService": ["Yes"],

    "MultipleLines": ["No"],

    "InternetService": ["Fiber optic"],

    "OnlineSecurity": ["No"],

    "OnlineBackup": ["Yes"],

    "DeviceProtection": ["No"],

    "TechSupport": ["No"],

    "StreamingTV": ["Yes"],

    "StreamingMovies": ["Yes"],

    "Contract": ["Month-to-month"],

    "PaperlessBilling": ["Yes"],

    "PaymentMethod": ["Electronic check"],

    "MonthlyCharges": [89.50],

    "TotalCharges": [1074.00]
})


# 4 . Prediction

prediction = model.predict(customer_data)

probability = model.predict_proba(customer_data)


# 5 . Result

print("\n Prediction Result")

if prediction[0] == 1:
    print("Customer will churn")
else:
    print("Customer will stay")

print(f"\n Churn Probability : {probability[0][1]:.2%}")


print(f"Retention Probability: {probability[0][0]:.2%}")

print("\n Step 7 completed successfully")