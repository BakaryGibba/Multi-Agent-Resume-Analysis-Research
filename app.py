import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

# Wait to load models until a request comes in so it doesn't crash on boot if the files are missing
xgb_model = None
scaler = None

def load_models_if_needed():
    global xgb_model, scaler
    model_path = 'best_resume_screening_model.pkl'
    scaler_path = 'scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        if xgb_model is None or scaler is None:
            print("Loading model and scaler...")
            xgb_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Loaded successfully!")
        return True
    return False

@app.route('/predict', methods=['POST'])
def predict():
    if not load_models_if_needed():
        return jsonify({
            "status": "error",
            "message": "Models not found! Please run the notebook and download the .pkl files into this directory first."
        }), 404

    # 1. Receive data from n8n (Extracted by Gemini)
    raw_data = request.json
    
    # Handle case where n8n sends stringified JSON instead of a raw object
    if isinstance(raw_data, str):
        data = json.loads(raw_data)
    else:
        data = raw_data

    # 2. Extract base features. 
    base_features = {
        'skills_match_score': float(data.get('skills_match_score', 50)),
        'project_count': float(data.get('project_count', 2)),
        'github_activity': float(data.get('github_activity', 20)),
        'years_experience': float(data.get('years_experience', 2)),
        'resume_length': float(data.get('resume_length', 300))
    }
    
    input_df = pd.DataFrame([base_features])

    # 3. APPLY YOUR NOVEL FEATURE ENGINEERING
    input_df['candidate_strength_index'] = (
        (input_df['skills_match_score'] * 0.5) +
        (input_df['project_count'] * 2) +
        (input_df['github_activity'] * 0.05)
    )
    input_df['experience_efficiency'] = (
        input_df['project_count'] / (input_df['years_experience'] + 1)
    )
    input_df['resume_quality_score'] = (
        input_df['resume_length'] * 0.2 +
        input_df['skills_match_score'] * 0.8
    )

    # 4. Align columns safely with what the scaler expects
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0 # Fallback for any other columns that were dropped
            
    input_df = input_df[expected_cols]

    # 5. Scale and Predict using XGBoost
    X_scaled = scaler.transform(input_df)
    probability = xgb_model.predict_proba(X_scaled)[0][1]
    prediction = xgb_model.predict(X_scaled)[0]

    # 6. Send result back to n8n
    return jsonify({
        "status": "success",
        "shortlisted_prediction": int(prediction),
        "match_probability": round(float(probability) * 100, 2), # Convert to percentage
        "features_used": data
    })

if __name__ == '__main__':
    print("Starting AI Resume Screening API...")
    print("Make sure 'best_resume_screening_model.pkl' and 'scaler.pkl' are in this folder!")
    # Trigger auto-reload
    app.run(host='0.0.0.0', port=5000, debug=True)
