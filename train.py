import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# --- STEP 2: TRAIN THE ARTIFICIAL INTELLIGENCE ---
# This script takes the raw profile data and turns it into a 'Brain' (Model) 
# that can recognize patterns between customer behavior and churn.

def train_churn_model():
    print("Loading Real-World IBM Telco Dataset...")
    data_path = 'data/telco_churn_real.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run generate_data.py first.")
        return

    # Load the CSV into pandas
    df = pd.read_csv(data_path)
    
    # 1. DATA CLEANING (Essential to prevent AI 'Confusion')
    # customerID is unique to every person; it has no predictive power, so we drop it.
    df = df.drop('customerID', axis=1)
    
    # In this dataset, some TotalCharges are empty strings (" ") instead of numbers.
    # We convert them to numeric, and 'coerce' turns those spaces into NaN (Not a Number).
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # We drop the few rows where TotalCharges was missing to keep our training pure.
    df = df.dropna()
    
    # The machine needs numbers, so we map 'Yes' to 1 (Churned) and 'No' to 0 (Stayed).
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Separate our Target (what we want to predict) from our Features (the data used to predict).
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 2. FEATURE GROUPING
    # Different types of data need different 'treatment'.
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # 3. PREPROCESSING PIPELINE (The 'Translation' Layer)
    # StandardScaler: Makes numbers (like 1,000 and 0.1) comparable.
    # OneHotEncoder: Turns categories (Male/Female) into binary columns the AI understands.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Split data: 80% for learning, 20% for 'testing' the AI's knowledge.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. MODEL COMPETITION
    # We test two famous algorithms to see which is the smartest for this specific data.
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    print("\nStarting Model Tournament...")
    for name, m in models.items():
        # Pipeline combines the translation layer (preprocessing) and the math layer (model)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', m)
        ])
        
        # Training the AI!
        pipeline.fit(X_train, y_train)
        
        # Testing the AI
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Results: {acc*100:.2f}% Accuracy")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = pipeline
            best_model_name = name

    print(f"\nTournament Winner: {best_model_name}")
    
    # 5. SAVING THE PRODUCTION ARTIFACTS
    # We 'pickle' (serialize) the entire pipeline into a file.
    # This allows the API (app.py) to load the intelligence instantly.
    print("Saving Production Artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/churn_model.joblib')
    
    # We also save the original feature list to ensure the API uses the same layout.
    joblib.dump(list(X.columns), 'models/features.joblib')
    print("Intelligence stored successfully in 'models/'. Ready for deployment.")

if __name__ == '__main__':
    train_churn_model()
