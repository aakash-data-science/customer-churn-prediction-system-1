# LOAD DATASET
# INDUSTRY LEVEL CUSTOMER CHURN PROJECT
# STEP 1

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (OneHotEncoder,StandardScaler)

from sklearn.model_selection import (train_test_split , cross_val_score , GridSearchCV)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ( RandomForestClassifier , GradientBoostingClassifier)

from sklearn.metrics import (accuracy_score , precision_score , recall_score , f1_score , confusion_matrix , classification_report , roc_curve , roc_auc_score)

import matplotlib.pyplot as plt

import seaborn as sns

import joblib

import shap

# 1 . Load Dataset 

df = pd.read_csv("C:\\Users\\aakas\\OneDrive\\DATA SCINCE PROJECTS\\customer_churn_project\\data\\raw\\project1.csv")

print("Data Loaded Successfully")

# 2 . Basic Info

print("\n First 5 Rows")
print(df.head())

print("\n Dataset Shape")
print(df.shape)

print("\n Missing Values")
print(df.isnull().sum())

# 3 . Remove Customer ID

df.drop("customerID", axis=1 , inplace=True)

print("\n Customer ID Removed")

# 4 . Fix Total Charges

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

# 5 .Handling Missing Values

df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\n MIssing Values After Cleaning")

print(df.isnull().sum())

# 6 . Target Analysis

print("\n Churn Distribution")

print(df["Churn"].value_counts())

print("\n Data Cleaning Completed")

#7 . Target Analysis
 
print("\n Churn Distribution") 

print(df["Churn"].value_counts())

# 8 . ENcode Target Variable

df["Churn"] = df["Churn"].map({"No" : 0,
                                "Yes" : 1})

print("\n Target Variable Encoded")

# 9 . Features and Target

X = df.drop( "Churn",axis=1)

y = df["Churn"]

# 10 . Identify Column Types

categorical_columns = X.select_dtypes( include = ["object" , "string"]).columns

numerical_columns = X.select_dtypes( include = ["int64" , "float64"]).columns



print("\n Categorical Columns")

print(categorical_columns)

print("\n Numerical Columns")
print(numerical_columns)

# 11 . Create Preprocessor

preprocess_data = ColumnTransformer( 
    transformers = [("num",StandardScaler(), numerical_columns),
                
                ("cat",OneHotEncoder(handle_unknown = "ignore"), categorical_columns)
                ]
)

print("Preprocessor completed succesfully")


# 12 . TRAIN TEST SPLIT

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size = 0.20 , stratify = y , random_state = 42)

print("train test split completed")

print("Training Shape :", X_train.shape)

print("Testing Shape :", X_test.shape)



# Model Dictionary

models = {
    
    "LogisticRegression" : 
    LogisticRegression(max_iter=1000),
          
    "Random Forest  " :
         RandomForestClassifier(
              n_estimators=100 ,
              random_state =42
        ),
          
          "Gradient Boosting ":
             GradientBoostingClassifier(
                 random_state=42
              
          )}

print("\n Models Loaded Successfully")

# Model Comparision

results =[]

best_model = None

best_model_name = ""

best_f1_score = 0

for model_name, model in models.items():
    print(f"\n Training {model_name}")

    pipeline = Pipeline([
                    ("preprocessor" , preprocess_data),
                    ("classifier", model)
                    ])
    
    print("\n Cross Validation Results")

    cv_scores = cross_val_score(

    pipeline,

    X_train,

    y_train,

    cv = 5,

    scoring = "f1"

)

    print("CV Scores :", cv_scores)

    print("Mean CV Score :", cv_scores.mean())

    print("Std CV Score :", cv_scores.std())
 

# 14 . Train Model 

    pipeline.fit(X_train , y_train)

#print("\n Model Training Completed")

# 15 . PREDICTIONS
    y_pred = pipeline.predict(X_test)

#print("\n Predictions Generated Successfully")

# 16 . EVALUATION

    accuracy = accuracy_score( y_test , y_pred)  #Accuracy asks: Out of all predictions, how many were correct?

    precision = precision_score( y_test , y_pred)

    recall = recall_score( y_test , y_pred)

    f1 = f1_score(y_test , y_pred)


    results.append({
    "Model" : model_name,
    "Accuracy" : accuracy,
    "Precision" : precision,
    "Recall" : recall,
    "F1 Score" : f1,
    "CV Mean Score" : cv_scores.mean()
    })


    print("\n Model Performance")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall : {recall:.4f}")

    print(f"F1 score : {f1:.4f}")


    if f1 > best_f1_score:
    
        best_f1_score = f1

        best_model = pipeline

        best_model_name = model_name

        best_predictions = y_pred

# Results table

results_df = pd.DataFrame(results)

results_df.to_csv("model_comparison_results.csv",
                  index=False)

print("Results Saved Successfully")

print("\n Model Comparision")

print(results_df)

print("\n Best Model")

print(best_model_name)

print(f"Best F1 Score : {best_f1_score:.4f}")

print("\n Starting Hyperparameter Tuning...")

param_grid = {

    "classifier__n_estimators": [100, 200],

    "classifier__learning_rate": [0.05, 0.1],

    "classifier__max_depth": [3, 5]

}

gb_pipeline = Pipeline([

    ("preprocessor", preprocess_data),

    ("classifier", GradientBoostingClassifier(
        random_state=42
    ))

])

grid_search = GridSearchCV(

    estimator = gb_pipeline,

    param_grid = param_grid,

    cv = 5,

    scoring = "f1",

    n_jobs = -1,

    verbose = 2

)

grid_search.fit(
    X_train,
    y_train
)


print("\n Best Parameters")

print(grid_search.best_params_)

print("\n Best CV Score")

print(grid_search.best_score_)

feature_importance = best_model.named_steps[ "classifier"].feature_importances_

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({

    "Feature" : feature_names,

    "Importance" : feature_importance
})

importance_df = importance_df.sort_values( by = "Importance", ascending = False)

importance_df.to_csv( "feature_importance.csv", index = False)

print("\n Top Features")

print(importance_df.head(12))



# 17 . CLASSIFICATION REPORT

print("\n Classification Report")

print(classification_report(y_test,best_predictions))

# ROC AUC SCORE

# ROC AUC SCORE

y_probabilities = best_model.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(
    y_test,
    y_probabilities
)

print("\n AUC Score")

print(f"AUC Score : {auc_score:.4f}")

# ROC CURVE

# ROC CURVE

fpr, tpr, thresholds = roc_curve(

    y_test,

    y_probabilities

)

plt.figure(figsize=(8,6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {auc_score:.4f}"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()

# 18 . CONFUSION MATRIX

cm = confusion_matrix( y_test , best_predictions)

sns.heatmap( cm , annot = True , fmt = "d" , cmap="Blues" )


plt.title(f"Confusion Matrix-{best_model_name}")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.show()

print("\n Step 5 Completed successfully")

# STEP 19 : SHAP EXPLAINER

sample_data = X_train.sample(
    100,
    random_state = 42
)

processed_sample = (
    best_model
    .named_steps["preprocessor"]
    .transform(sample_data)
)

explainer = shap.TreeExplainer(

    best_model.named_steps["classifier"]
)

joblib.dump(

    explainer,

    r"C:\Users\aakas\OneDrive\DATA SCINCE PROJECTS\customer_churn_project\models\shap_explainer.joblib"
)

print(
    "\n SHAP Explainer Saved Successfully"
)

# 19 . JOBLIB

joblib.dump(best_model,r"C:\Users\aakas\OneDrive\DATA SCINCE PROJECTS\customer_churn_project\models\customer_churn_model.joblib")

print("\n Model Saved Successfully")

print("Training pipeline completed")


# important notes use of joblib and how pipeline works 

# It only stores the learned preprocessing information and the learned model patterns needed for prediction.'''

#Training is the process where a machine learning algorithm analyzes historical data and learns patterns that can later be used to make predictions.'''

# We save the trained model using joblib so we don't have to repeat the training process every time we want a prediction.  "'''

# "Training the model requires analyzing the dataset and learning patterns, which takes time. We save the trained model using joblib so that we can load it later 
 
# and make predictions instantly without retraining." '''

# Saves best_predictions and uses them for the Classification Report and Confusion Matrix.