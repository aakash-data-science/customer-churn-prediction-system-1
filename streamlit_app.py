# STREAMLIT APP
# STEP 8

# STREAMLIT APP
# STEP 8

from reportlab.pdfgen import canvas

from io import BytesIO

import streamlit as st

import pandas as pd

import joblib

import shap

import sqlite3

from datetime import datetime

# STEP 21 : Database Module

from database import *

import matplotlib.pyplot as plt


import plotly.express as px

if "batch_results" not in st.session_state:

    st.session_state.batch_results = None

if "single_prediction" not in st.session_state:

    st.session_state.single_prediction = None

if "uploaded_batch_df" not in st.session_state:

    st.session_state.uploaded_batch_df = None

# Load Model

model = joblib.load(r"models/customer_churn_model.joblib")

# STEP 20 : SQLITE DATABASE


#import os

#os.makedirs("database", exist_ok=True)

#conn = sqlite3.connect(
 #   "database/churn_history.db",
  #  check_same_thread=False
#)
#cursor = conn.cursor()


#cursor.execute(

#'''

#CREATE TABLE IF NOT EXISTS prediction_history(

#id INTEGER PRIMARY KEY AUTOINCREMENT,

#prediction TEXT,

#churn_probability REAL,

#retention_probability REAL,

#risk_level TEXT,

#timestamp TEXT

#)

#'''
#)


#conn.commit()
#'''

# STEP 19 : Load SHAP Explainer

explainer = joblib.load(
    r"models/shap_explainer.joblib"
    )

# page config

st.set_page_config(
    page_title = "Customer Churn Analysis Platform",
    page_icon = "📊",
    layout = "wide"
)

# Page Title 

st.title("📊 Customer Churn Prediction System")

st.markdown("Predict whether a customer is likely to churn using Machine Learning.")

st.markdown("---")

#st.subheader("Enter Customer Information")

# Inputs

st.subheader("Enter Customer Information")

col1 , col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Male" , "Female"])

    senior_citizen = st.selectbox("Senior Citizen", [0 ,1])

    partner = st.selectbox("Partner" , ["Yes","No"])

    dependents = st.selectbox("Dependents",["Yes", "No"])

    tenure = st.number_input("Tenure", min_value =0 ,
                             max_value = 100,
                             value=12)

with col2:

    phone_service = st.selectbox( "Phone Service" , ["Yes" , "No"])

    internet_service = st.selectbox("Internet Service",["DSL" , "Fiber optic" , "No"])

    contract = st.selectbox( "Contract" , ["Month-to-month" , "One year" , "Two year"])

    payment_method = st.selectbox( "Payment Method" , [
                                    "Electronic check" , 
                                    "Mailed check",
                                    "Bank transfer (automatic)",
                                    "Credit card (automatic)"])

    monthly_charges = st.number_input("Monthly Charges", min_value = 0.0,
                                  value = 89.50)

# Prediction Button


if st.button("Predict Churn"):

    customer_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": ["No"],

        "InternetService": [internet_service],

        "OnlineSecurity": ["No"],

        "OnlineBackup": ["Yes"],

        "DeviceProtection": ["No"],

        "TechSupport": ["No"],

        "StreamingTV": ["Yes"],

        "StreamingMovies": ["Yes"],

        "Contract": [contract],

        "PaperlessBilling": ["Yes"],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [
            tenure * monthly_charges
        ]
    })

    prediction = model.predict(customer_data)

    probability = model.predict_proba(customer_data)

    # STEP 19.6 : SHAP Data Preparation

    processed_customer = (

    model.named_steps["preprocessor"].transform(customer_data)
    )

    feature_names = (

    model.named_steps["preprocessor"].get_feature_names_out()
    )

    # STEP 19.7 : SHAP Values

    shap_values = explainer.shap_values(

    processed_customer
    )

    # STEP 19.8 : SHAP Feature Contribution

    shap_df = pd.DataFrame({

    "Feature" : feature_names,

    "Impact" : abs(shap_values[0])

    })

    shap_df = shap_df.sort_values(

    by = "Impact",

    ascending = False
    )


    # top_shap_features = shap_df.head(10)

    shap_df["Feature"] = (

    shap_df["Feature"]

    .str.replace("cat__", "", regex=False)

    .str.replace("num__", "", regex=False)

    .str.replace("_", " ", regex=False)
)

    top_shap_features = shap_df.head(10)

    churn_probability = probability[0][1]

    retention_probability = probability[0][0]

    if churn_probability >= 0.75:

        risk_level = "High"

    elif churn_probability >= 0.50:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    st.session_state.single_prediction = {

        "prediction": prediction[0],

        "churn_probability": churn_probability,

        "retention_probability": retention_probability,

        "risk_level": risk_level
    }

    # STEP 20.1 : Save Prediction


    cursor.execute(

    '''

    INSERT INTO prediction_history(

    prediction,

    churn_probability,

    retention_probability,

    risk_level,

    timestamp

    )

    VALUES(?,?,?,?,?)

    ''',


    (

    "Will Churn"

    if prediction[0]==1

    else

        "Will Stay",


    float(churn_probability),

    float(retention_probability),

    risk_level,


    datetime.now()

    .strftime(

    "%d-%m-%Y %H:%M:%S"

    )

    )

    )

    conn.commit()

    # STEP 20.1.1 : Debug SQLite Saving

    cursor.execute(

    '''

    SELECT COUNT(*)

    FROM prediction_history

    '''

    )

    count = cursor.fetchone()[0]

    st.write(

    f"DEBUG : Rows in SQLite = {count}"

    )

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(" ⚠️ Customer Will Churn")
        
    else:

        st.success(" ✅ Customer Will Stay")

    col1 , col2 , col3 = st.columns(3)

    with col1:

        st.metric(
            label = "Churn Probability",
            value = f"{churn_probability:.2%}"
        )


    with col2:
        st.metric(

            label= "Retention Probability",
            value = f"{retention_probability:.2%}"
        
        )
    
    with col3:

        
        st.metric(
            label = "Risk Level",
            value = risk_level
        )

    # STEP 19.9 : SHAP Explainability

    st.markdown("---")

    st.subheader(
        "Why This Customer Received This Prediction"
        )

    st.dataframe(

    top_shap_features,

        width = "stretch"
        )


   # st.success("Button Working Successfully")
    # PDF Executive Report

    pdf_buffer = BytesIO()

    pdf = canvas.Canvas(pdf_buffer)

    pdf.setTitle("Customer Churn Report")

    pdf.drawString(
    100,
    800,
    "Customer Churn Prediction Report"
    )

    pdf.drawString(
    100,
    760,
    f"Prediction : {'Will Churn' if prediction[0] == 1 else 'Will Stay'}"
    )

    pdf.drawString(
    100,
    730,
    f"Churn Probability : {churn_probability:.2%}"
    )

    pdf.drawString(
    100,
    700,
    f"Retention Probability : {retention_probability:.2%}"
    )

    pdf.drawString(
    100,
    670,
    f"Risk Level : {risk_level}"
    )

    pdf.save()

    pdf_buffer.seek(0)


    st.subheader(
        "Risk Analysis"
    )

    if churn_probability >= 0.75:

        st.error(
            "🔴 HIGH CHURN RISK"
        )

        st.subheader("Recommended Business Actions")
        
        st.write("• Offer customer retention discount")

        st.write("• Assign dedicated support representative")

        st.write("• Schedule immediate follow-up call")

        st.write("• Provide loyalty rewards program")

    elif churn_probability >= 0.50:

        st.warning(
            "🟡 MEDIUM CHURN RISK"
        )
        st.subheader(
        "Recommended Business Actions"
        )

        st.write(
        "• Send promotional offers"
        )

        st.write(
        "• Encourage annual contract upgrade"
        )

        st.write(
        "• Monitor customer activity"
        )

        st.write(
        "• Increase engagement campaigns"
         )

    else:

        st.success(
            "🟢 LOW CHURN RISK"
        )
        st.subheader(
        "Recommended Business Actions"
        )

        st.write(
        "• Maintain current customer experience"
        )

        st.write(
        "• Continue regular engagement"
        )

        st.write(
        "• Offer optional premium services"
        )

        st.write(
        "• Monitor satisfaction periodically"
        )


        # Download Report

    report_df = pd.DataFrame({

            "Prediction" : [
             "Will Churn" if prediction[0] == 1 else "Will Stay"
              ],

            "Churn Probability" : [
                f"{churn_probability:.2%}"
                ],

             "Retention Probability" :[
              f"{retention_probability:.2%}"
                   ],

             "Risk Level" : [
                 risk_level
                  ]
                })

    st.download_button(

            label = "📥 Download Prediction Report",

            data = report_df.to_csv(index = False),

            file_name = "prediction_report.csv",

             mime = "text/csv",

            on_click="ignore"

            )
    
    st.download_button(

    label = "📄 Download Executive PDF Report",

    data = pdf_buffer,

    file_name = "executive_report.pdf",

    mime = "application/pdf",

    on_click="ignore"
        )
    # Persistent Prediction Result


# Model Information

st.markdown("---")

st.subheader("Model Information")

st.write("Best Model : Gradient Boosting")

st.write("Selection Metric : F1 Score")

st.write("Purpose : Customer Churn Prediction")

# Feature Importance

st.markdown("---")

st.subheader("Top Churn Factors")

importance_df = pd.read_csv("feature_importance.csv")

importance_df["Feature"]=(importance_df["Feature"]

    .str.replace("cat__","",regex=False)

    .str.replace("num__","",regex=False)

    .str.replace("_"," ",regex =False)
)


importance_df["Feature"] = importance_df["Feature"].replace({

    "tenure": "Tenure",

    "MonthlyCharges": "Monthly Charges",

    "TotalCharges": "Total Charges",

    "InternetService Fiber optic":
        "Internet Service (Fiber Optic)",

    "InternetService DSL":
        "Internet Service (DSL)",

    "PaymentMethod Electronic check":
        "Payment Method (Electronic Check)",

    "Contract Month-to-month":
        "Contract (Month-to-Month)",

    "Contract One year":
        "Contract (One Year)",

    "Contract Two year":
        "Contract (Two Year)"
})

# select Top 10 Features

top_features = importance_df.head(12)

# create chart

fig = px.bar(

    top_features,

    x= "Importance",

    y = "Feature",

    orientation = "h",

    title = "Top 12 Feature Importance Score"


)

fig.update_layout(

    height = 600,

    yaxis = dict(

        autorange = "reversed"
    )
)

st.plotly_chart(

    fig,

    width = "stretch"
)

# Model Comparision Table

st.markdown("---")

st.subheader("Model Comparison")

results_df = pd.read_csv("model_comparison_results.csv")

st.dataframe(results_df , width = "stretch")

# Confusion Matrix

st.markdown("---")

st.subheader("Confusion Matrix")

st.image(
    "confusion_matrix.png" , width = "stretch"
)

# ROC Curve

st.markdown("---")

st.subheader("ROC Curve")

st.image(
    "roc_curve.png" , width = "stretch"
)


# Batch Csv Prediction
# Batch Csv Prediction

st.markdown("---")

st.subheader("Batch Customer Prediction")

uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.session_state.uploaded_batch_df = batch_df

    st.success(
        "CSV File Uploaded Successfully"
    )

    st.write(
        "Preview of Uploaded Data"
    )

    st.dataframe(
        batch_df.head()
    )

    if st.button(
        "Predict All Customers"
    ):

        predictions = model.predict(
            batch_df
        )

        probabilities = model.predict_proba(
            batch_df
        )

        batch_df["Prediction"] = predictions

        batch_df["Churn Probability"] = probabilities[:,1]

        batch_df["Retention Probability"] = probabilities[:,0]

        st.session_state.batch_results = batch_df


# Batch Prediction Results

if st.session_state.batch_results is not None:

    batch_df = st.session_state.batch_results

    st.success(
    "Batch Prediction Completed"
    )

# Business KPI Dashboard

    st.markdown("---")

    st.subheader("Business KPI Dashboard")

    col1 , col2 , col3 , col4 = st.columns(4)

    estimated_revenue_loss = (

    batch_df.loc[
        batch_df["Prediction"] == 1,
        "MonthlyCharges"
            ].sum() * 12
        )

    with col1:

        st.metric(
        "Total Customers",
        len(batch_df)
        )

    with col2:

     st.metric(
        "High Risk Customers",
        (batch_df["Churn Probability"] >= 0.75).sum()
     )

    with col3:

     st.metric(
        "Average Churn Risk",
        f"{batch_df['Churn Probability'].mean()*100:.2f}%"
        )
    with col4:

        st.metric(

        "Estimated Revenue Loss",

        f"${estimated_revenue_loss:,.0f}"
        )
     
      

    st.subheader(
    "Predicted Customer Records"
        )

    st.dataframe(
    batch_df.head(20),
    width="stretch"
    )
    
    total_customers = len(
        batch_df
    )

    churn_customers = (
        batch_df["Prediction"] == 1
    ).sum()

    stay_customers = (
        batch_df["Prediction"] == 0
    ).sum()

    avg_churn_probability = (

        batch_df[
            "Churn Probability"
        ].mean() * 100
    )

    # Batch Prediction Summary

    st.subheader(
        "Batch Prediction Summary"
    )

    col1 , col2 , col3 , col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            total_customers
        )

    with col2:

        st.metric(
            "Will Churn",
            churn_customers
        )

    with col3:

        st.metric(
            "Will Stay",
            stay_customers
        )

    with col4:

        st.metric(
        "Average Churn Risk",
        f"{avg_churn_probability:.2f}%"
    )

    # STEP 14 : Customer Risk Segmentation

    st.markdown("---")

    high_risk = (
    batch_df["Churn Probability"] >= 0.75
    ).sum()

    medium_risk = (
    (batch_df["Churn Probability"] >= 0.50)
    &
    (batch_df["Churn Probability"] < 0.75)
    ).sum()

    low_risk = (
    batch_df["Churn Probability"] < 0.50
    ).sum()

    risk_df = pd.DataFrame({

    "Risk Category":[
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ],

    "Customers":[
        high_risk,
        medium_risk,
        low_risk
    ]
    })

    # STEP 15 : Top High Risk Customers

    st.markdown("---")

    st.subheader(
    "Top 20 High Risk Customers"
        )

    top_risk_customers = (

    batch_df[
        batch_df["Churn Probability"] >= 0.75
    ]

    .sort_values(

        by="Churn Probability",

        ascending=False

    )

    .head(20)
    )

    # STEP 17 : High Risk Customer Count

    st.metric(

    "Top Risk Customers Listed",

    len(top_risk_customers)
    )

    st.dataframe(

    top_risk_customers,

    width = "stretch"
    )

    # STEP 16 : Download High Risk Customers

    high_risk_csv = top_risk_customers.to_csv(
    index=False
    )

    st.download_button(

    label="📥 Download Top Risk Customers",

    data=high_risk_csv,

    file_name="top_risk_customers.csv",

    mime="text/csv",

    on_click="ignore"
    )

    # STEP 14.1 : Risk Segmentation Chart

    risk_fig = px.pie(

    risk_df,

    names = "Risk Category",

    values = "Customers",

    title = "Customer Risk Segmentation",

    hole = 0.4
    )

    risk_fig.update_traces(

    textposition = "outside",

    textinfo = "label+percent",

    textfont_size = 20
    )

    risk_fig.update_layout(

    height = 550,

    title_x = 0.5,

    title_font = dict(size = 24),

    legend = dict(

        orientation = "h",

        yanchor = "bottom",

        y = -0.15,

        xanchor = "center",

        x = 0.5,

        font = dict(size = 16)
    )
    )

    st.plotly_chart(
    risk_fig,
    width="stretch"
    )

    # STEP 14.5 : Download Risk Segmentation Chart

    
    # STEP 14.6 : Batch Results CSV

    csv = batch_df.to_csv(
    index=False
    )
    st.download_button(

        label="📥 Download Batch Results",

        data=csv,

        file_name="batch_predictions.csv",

        mime="text/csv",

        on_click="ignore"
    )


    
# STEP 20.2 : Prediction History Dashboard


st.markdown("---")

st.subheader("Prediction History")

history = pd.read_sql(

    '''
    SELECT *

    FROM prediction_history

    ORDER BY id DESC

    ''',

    conn

    )


# STEP 20.2.1 : Beautify Column Names


history = history.rename(

    columns={

    'id':'ID',

    'prediction':'Prediction',

    'churn_probability':'Churn Probability',

    'retention_probability':'Retention Probability',

    'risk_level':'Risk Level',

    'timestamp':'Timestamp'

    }

    )


# STEP 20.2.2 : History Metrics


col1, col2, col3, col4= st.columns(4)

with col1:

        st.metric(

        "Total Predictions",

        len(history)

    )

with col2:

        st.metric(

        "High Risk",

        (

            history['Risk Level']

            == 'High'

        ).sum()

        )

with col3:

        st.metric(

        "Medium Risk",

        (

            history['Risk Level']

            == 'Medium'

        ).sum()

        )
with col4:

    st.metric(

        "Low Risk",

        (

            history['Risk Level']

            == 'Low'

        ).sum()

    )

# STEP 20.2.3 : Display Table


st.dataframe(

    history,

    width="stretch"

    )

    

    ###########################################################
# STEP 20.3 : Download Prediction History
###########################################################

history_csv = history.to_csv(

    index = False

    )


st.download_button(

    label = "📥 Download Prediction History",

    data = history_csv,

    file_name = "prediction_history.csv",

    mime = "text/csv",

    on_click = "ignore"

    )

###########################################################
# STEP 20.4 : Clear Prediction History
###########################################################

if st.button(

    "🗑 Clear History"):


        cursor.execute(

        '''

        DELETE FROM prediction_history

        '''

     )


        conn.commit()


        st.rerun()


# Footer

st.markdown("---")

st.caption("Built using Scikit-Learn , Streamlit and Machine Learning")
