# Customer Churn Prediction System

## Project Overview

This project predicts whether a telecom customer is likely to churn using Machine Learning.

The system helps businesses identify high-risk customers and take proactive retention actions.

---

## Business Problem

Customer churn causes revenue loss for telecom companies.

This project analyzes customer information and predicts churn probability, allowing businesses to improve customer retention.

---

## Dataset

Dataset Size: 120,000 Customers

Features Include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Contract Type
- Internet Service
- Payment Method
- Monthly Charges
- Total Charges

Target Variable:

- Churn (Yes / No)

---

## Machine Learning Pipeline

1. Data Cleaning
2. Feature Engineering
3. Preprocessing Pipeline
4. Train-Test Split
5. Model Training
6. Cross Validation
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Deployment

---

## Models Compared

- Logistic Regression
- Random Forest
- Gradient Boosting

Best Model:

Gradient Boosting

---

## Model Performance

Accuracy: 72.37%

Precision: 71.24%

Recall: 81.34%

F1 Score: 75.95%

AUC Score: 78.39%

---

## Features

### Single Customer Prediction

Predict customer churn probability.

### Risk Analysis

- Low Risk
- Medium Risk
- High Risk

### Business Recommendations

Provides retention actions based on churn risk.

### Batch Prediction

Upload CSV files containing customer data and predict churn for thousands of customers.

### Download Reports

Download prediction results as CSV files.

### Interactive Dashboard

Built using Streamlit and Plotly.

---

## Technology Stack

Python

Pandas

NumPy

Scikit-Learn

Streamlit

Plotly

Matplotlib

Joblib

---

## Project Structure

customer_churn_project/

├── data/

├── models/

├── src/

│   ├── train.py

│   └── predict.py

├── streamlit_app.py

├── README.md

└── requirements.txt

---

## Future Improvements

- SHAP Explainability
- Cloud Deployment
- Real-Time Predictions
- Automated Model Retraining

---

## Author 

  Aakash