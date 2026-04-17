from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# --- STEP 3: DEPLOY THE WEB SERVICE ---
# This script starts a local web server (API). 
# It connects the Machine Learning 'Brain' to the Frontend dashboard.

app = Flask(__name__)

# File paths where our trained intelligence is stored
MODEL_PATH = 'models/churn_model.joblib'
FEATURES_PATH = 'models/features.joblib'

model = None
features = []

# This function runs as soon as we start the server
def load_environment():
    global model, features
    if os.path.exists(MODEL_PATH):
        # We load the entire Preprocessing + Model pipeline
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        print("Intelligence successfully loaded from storage.")

load_environment()

# Route: The homepage that serves our beautiful UI
@app.route('/')
def home():
    return render_template('index.html')

# Route: The brain of the API. It receives data from the frontend and returns a prediction.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. RECEIVE DATA: Get the JSON object sent by the UI form
        data = request.json
        if not model or not features:
            return jsonify({'error': 'Server Error: Production model not loaded.'}), 500
        
        # 2. FORMAT DATA: Convert the single customer JSON into a Pandas DataFrame
        # The pipeline expects a DataFrame structure to match the training data.
        input_df = pd.DataFrame([data])
        
        # 3. VALIDATE: Ensure the frontend sent every field the AI expects
        missing = [f for f in features if f not in input_df.columns]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
            
        # 4. PREDICT: Pass the DataFrame through the pipeline
        # The pipeline automatically handles scaling and encoding before calculating the risk.
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        # 5. RETURN: Send the results back to the UI in a friendly JSON format
        result = {
            'churn_prediction': 'Yes' if int(prediction) == 1 else 'No',
            'churn_probability': float(prob[1]), 
            'stay_probability': float(prob[0])
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the server on port 5000 (localhost)
    app.run(debug=False, port=5000)
