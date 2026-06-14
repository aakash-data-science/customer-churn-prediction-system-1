# STREAMLIT APP
# STEP 8


import streamlit as st

import pandas as pd

import joblib

import matplotlib.pyplot as plt

import plotly.express as px
# Load Model

model = joblib.load(r"customer_churn_model.joblib")

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

    churn_probability = probability[0][1]

    retention_probability = probability[0][0]

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

        if churn_probability >= 0.75:

            risk_level = "High"

        elif churn_probability >= 0.50:

            risk_level = "Medium"

        else:

            risk_level = "Low"

        st.metric(
            label = "Risk Level",
            value = risk_level
        )

   # st.success("Button Working Successfully")
    
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

             mime = "text/csv"

            )

        



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

    use_container_width = True
)

# Model Comparision Table

st.markdown("---")

st.subheader("Model Comparison")

results_df = pd.read_csv("model_comparison_results.csv")

st.dataframe(results_df , width = "stretch")

# Batch Csv Prediction

st.markdown("---")

st.subheader("Batch Customer Prediction")

uploaded_file = st.file_uploader("Upload Customer CSV File",
                               type = ["csv"]
                               )

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.success("CSV File Uploaded Successfully")

    st.write("Preview of Uploaded Data")

    st.dataframe(batch_df.head())

    if st.button("Predict All Customers"):
           

        predictions = model.predict(batch_df)
           
        probabilities = model.predict_proba(batch_df)
           
        batch_df["Prediction"] = predictions
           
        batch_df["Churn Probability"] = probabilities[:,1]
           
        batch_df["Retention Probability"] = probabilities[:,0]
           
        st.success("Batch Prediction Completed")
           
        st.dataframe(batch_df.head())

        total_customers = len(batch_df)

        churn_customers = (batch_df["Prediction"]==1).sum()

        stay_customers = (batch_df["Prediction"]==0).sum()

        avg_churn_probability = (
               
             batch_df["Churn Probability"].mean()*100
           )

        st.subheader("Batch Prediction Summary")

        col1 , col2 , col3 , col4 = st.columns(4)

        with col1:
            st.metric("Total Customers",
                         total_customers)
               
        with col2:
             st.metric("Will Churn",
                         churn_customers)
               
        with col3:
             st.metric("Will Stay",
                         stay_customers)
        with col4:
            st.metric("Average Churn Risk",
                         
                         f"{avg_churn_probability:.2f}%")
               

    #Customer Distribution Chart

        summary_df = pd.DataFrame({
             "Category": ["Will Churn", "Will Stay"],
             "Count": [churn_customers, stay_customers]
                })

        fig = px.pie(
            summary_df,
            names="Category",
            values="Count",
            title="Customer Distribution",
            hole=0.4
            )

        st.plotly_chart(fig, use_container_width=True)
               
        csv = batch_df.to_csv(index=False)

        st.download_button(
                label="📥 Download Batch Results",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
                  )       

# Footer

st.markdown("---")

st.caption("Built using Scikit-Learn , Streamlit and Machine Learning")
