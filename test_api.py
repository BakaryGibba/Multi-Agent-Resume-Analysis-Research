import requests
import json

url = 'http://127.0.0.1:5000/predict'

# This matches the JSON structure that Gemini will extract in Agent 1
test_data = {
    "skills_match_score": 85,
    "project_count": 4,
    "github_activity": 70,
    "years_experience": 3,
    "resume_length": 350
}

print(f"Sending test data to {url}...")
print(f"Payload: {json.dumps(test_data, indent=2)}\n")

try:
    response = requests.post(url, json=test_data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON from your XGBoost Model:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.ConnectionError:
    print("ERROR: Connection refused. Please make sure app.py is running!")
except Exception as e:
    print(f"An error occurred: {e}")
