import pandas as pd
import joblib
from flask import Flask, request, jsonify
import json
from pathlib import Path

app = Flask(__name__)

# Wait to load models until a request comes in so it doesn't crash on boot if the files are missing
xgb_model = None
scaler = None
BASE_DIR = Path(__file__).resolve().parent

BASE_FEATURES = (
    'skills_match_score',
    'project_count',
    'github_activity',
    'years_experience',
    'resume_length',
)
EDUCATION_LEVELS = {
    'bachelors': 0,
    'high school': 1,
    'masters': 2,
    'phd': 3,
}

def load_models_if_needed():
    global xgb_model, scaler
    model_path = BASE_DIR / 'best_resume_screening_model.pkl'
    scaler_path = BASE_DIR / 'scaler.pkl'
    
    if model_path.exists() and scaler_path.exists():
        if xgb_model is None or scaler is None:
            print("Loading model and scaler...")
            xgb_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Loaded successfully!")
        return True
    return False


def build_feature_frame(data):
    if not isinstance(data, dict):
        raise ValueError('The request body must be a JSON object.')

    required = (*BASE_FEATURES, 'education_level')
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    try:
        base_features = {name: float(data[name]) for name in BASE_FEATURES}
    except (TypeError, ValueError) as exc:
        raise ValueError('All feature values must be numeric.') from exc

    if any(value < 0 for value in base_features.values()):
        raise ValueError('Feature values must be non-negative.')

    education_level = str(data['education_level']).strip().lower()
    if education_level not in EDUCATION_LEVELS:
        allowed = ', '.join(EDUCATION_LEVELS)
        raise ValueError(f'education_level must be one of: {allowed}.')

    input_df = pd.DataFrame([base_features])
    input_df['education_level_encoded'] = EDUCATION_LEVELS[education_level]
    input_df['candidate_strength_index'] = (
        (input_df['skills_match_score'] * 0.5)
        + (input_df['project_count'] * 2)
        + (input_df['github_activity'] * 0.05)
    )
    input_df['experience_efficiency'] = (
        input_df['project_count'] / (input_df['years_experience'] + 1)
    )
    input_df['resume_quality_score'] = (
        input_df['resume_length'] * 0.2
        + input_df['skills_match_score'] * 0.8
    )
    return input_df


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_files_available': (
            (BASE_DIR / 'best_resume_screening_model.pkl').exists()
            and (BASE_DIR / 'scaler.pkl').exists()
        ),
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not load_models_if_needed():
        return jsonify({
            "status": "error",
            "message": "Models not found! Please run the notebook and download the .pkl files into this directory first."
        }), 404

    # Receive the constrained feature payload extracted by the LLM workflow.
    raw_data = request.get_json(silent=True)
    
    # Handle case where n8n sends stringified JSON instead of a raw object
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        input_df = build_feature_frame(data)
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400

    # 4. Align columns safely with what the scaler expects
    expected_cols = list(scaler.feature_names_in_)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    input_df = input_df[expected_cols]

    # The saved XGBoost model was trained on the unscaled feature frame. The
    # scaler is retained for experiment reproducibility and column metadata.
    probability = xgb_model.predict_proba(input_df)[0][1]
    prediction = xgb_model.predict(input_df)[0]

    # 6. Send result back to n8n
    return jsonify({
        "status": "success",
        "shortlisted_prediction": int(prediction),
        "match_probability": round(float(probability) * 100, 2), # Convert to percentage
        "features_used": data,
        "decision_support_only": True
    })

if __name__ == '__main__':
    print("Starting AI Resume Screening API...")
    print("Make sure 'best_resume_screening_model.pkl' and 'scaler.pkl' are in this folder!")
    # Trigger auto-reload
    app.run(host='127.0.0.1', port=5000, debug=False)
