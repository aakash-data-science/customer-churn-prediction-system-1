
 # STEP 19 : FASTAPI

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

import joblib


# STEP 19.1 : Load Model

model = joblib.load(

    r"models/customer_churn_model.joblib"

)


# STEP 19.2 : Initialize API

app = FastAPI(

    title="Customer Churn API",

    version="1.0"
)

###########################################################
# STEP 22 : Customer Request Model
###########################################################

class Customer(BaseModel):

    gender : str

    SeniorCitizen : int

    Partner : str

    Dependents : str

    tenure : int

    PhoneService : str


    MultipleLines : str = "No"


    InternetService : str


    OnlineSecurity : str = "No"

    OnlineBackup : str = "Yes"

    DeviceProtection : str = "No"

    TechSupport : str = "No"

    StreamingTV : str = "Yes"

    StreamingMovies : str = "Yes"



    Contract : str


    PaperlessBilling : str = "Yes"



    PaymentMethod : str


    MonthlyCharges : float

# STEP 19.3 : Home Route

@app.get("/")

def home():

    return {

        "Message":

        "Customer Churn Prediction API"

    }


# STEP 19.4 : Prediction Route


@app.post("/predict")

def predict_customer(

customer: Customer

):


    customer_data = pd.DataFrame({

'gender':[customer.gender],

'SeniorCitizen':[customer.SeniorCitizen],

'Partner':[customer.Partner],

'Dependents':[customer.Dependents],

'tenure':[customer.tenure],

'PhoneService':[customer.PhoneService],

'MultipleLines':[customer.MultipleLines],

'InternetService':[customer.InternetService],

'OnlineSecurity':[customer.OnlineSecurity],

'OnlineBackup':[customer.OnlineBackup],

'DeviceProtection':[customer.DeviceProtection],

'TechSupport':[customer.TechSupport],

'StreamingTV':[customer.StreamingTV],

'StreamingMovies':[customer.StreamingMovies],

'Contract':[customer.Contract],

'PaperlessBilling':[customer.PaperlessBilling],

'PaymentMethod':[customer.PaymentMethod],

'MonthlyCharges':[customer.MonthlyCharges],

'TotalCharges':[

customer.tenure * customer.MonthlyCharges

]

})
    
    ###########################################################
# STEP 22.1 : Model Prediction
###########################################################

    prediction = model.predict(

    customer_data

)


    probability = model.predict_proba(

    customer_data

    )

    ###########################################################
# STEP 22.2 : Risk Level
###########################################################

    risk_level = (

    "High"

    if probability[0][1] >= 0.75

    else

    "Medium"

    if probability[0][1] >= 0.50

    else

    "Low"

    )



    return {

    "Prediction":

    "Will Churn"

    if prediction[0] == 1

    else

    "Will Stay",


    "Risk Level":

    risk_level,


    "Churn Probability":

    round(

    float(probability[0][1]),

    4

    ),


    "Retention Probability":

    round(

    float(probability[0][0]),

    4

    )

    }
