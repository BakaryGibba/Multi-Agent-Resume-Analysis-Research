from unittest.mock import Mock

import numpy as np

import app as api


VALID_PAYLOAD = {
    'skills_match_score': 85,
    'project_count': 4,
    'github_activity': 70,
    'years_experience': 3,
    'resume_length': 350,
    'education_level': 'Masters',
}


def test_health_endpoint():
    response = api.app.test_client().get('/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'


def test_feature_engineering_is_deterministic():
    frame = api.build_feature_frame(VALID_PAYLOAD)
    assert frame.loc[0, 'candidate_strength_index'] == 54.0
    assert frame.loc[0, 'experience_efficiency'] == 1.0
    assert frame.loc[0, 'resume_quality_score'] == 138.0


def test_predict_rejects_missing_fields(monkeypatch):
    monkeypatch.setattr(api, 'load_models_if_needed', lambda: True)
    response = api.app.test_client().post('/predict', json={'project_count': 2})
    assert response.status_code == 400
    assert 'Missing required fields' in response.get_json()['message']


def test_predict_uses_unscaled_feature_frame(monkeypatch):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    model.predict.return_value = np.array([1])
    scaler = Mock()
    scaler.feature_names_in_ = np.array([
        'skills_match_score', 'project_count', 'github_activity',
        'years_experience', 'resume_length', 'candidate_strength_index',
        'experience_efficiency', 'resume_quality_score',
        'education_level_encoded',
    ])

    monkeypatch.setattr(api, 'xgb_model', model)
    monkeypatch.setattr(api, 'scaler', scaler)
    monkeypatch.setattr(api, 'load_models_if_needed', lambda: True)

    response = api.app.test_client().post('/predict', json=VALID_PAYLOAD)
    body = response.get_json()

    assert response.status_code == 200
    assert body['match_probability'] == 80.0
    assert body['decision_support_only'] is True
    assert not scaler.transform.called
    assert list(model.predict.call_args.args[0].columns) == list(scaler.feature_names_in_)
