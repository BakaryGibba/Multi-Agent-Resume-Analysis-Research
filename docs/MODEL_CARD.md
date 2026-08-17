# Model Card

[Back to project overview](../README.md)

## Model Summary

The selected predictor is an `XGBClassifier` trained on structured resume-job features and three engineered composites. It is exposed through a local Flask API and orchestrated between two LLM stages in n8n.

| Property | Recorded value |
| --- | --- |
| Task | Binary classification |
| Selected model | XGBoost |
| Training records | 24,000 |
| Test records | 6,000 |
| Split | Stratified 80/20 |
| Random seed | 42 |
| Accuracy | 0.9063 |
| Precision | 0.9289 |
| Recall | 0.9378 |
| F1 | 0.9333 |
| ROC-AUC | 0.9659 |

Metrics are copied from the committed notebook output. They have not been independently reproduced on an external dataset.

## Inputs

The API requires skills-match score, project count, GitHub activity, years of experience, resume length, and education level. It then derives:

- **Candidate Strength Index:** weighted skills, project, and GitHub signals.
- **Experience Efficiency:** projects divided by years of experience plus one.
- **Resume Quality Score:** weighted resume length and skills-match score.

These formulas are research hypotheses, not validated measures of candidate quality.

## Intended Use

- Educational research on modular AI pipelines.
- Comparison of classical classifiers on a public benchmark.
- Prototyping explanation and human-review interfaces.

## Out-Of-Scope Use

- Automated rejection, shortlisting, ranking, or hiring of real people.
- Inferring competence, character, employability, or protected traits.
- Monitoring employees or scraping candidate data without consent.
- Production deployment without fairness, privacy, security, and legal review.

## Limitations

- Evaluation uses one split from one dataset; no cross-validation or external validation is reported.
- Model selection is based on the same held-out split used for reported comparison.
- Probability calibration was not evaluated.
- The education encoding is ordinal only by implementation and should not be interpreted as an intrinsic hierarchy.
- The LLM explanation describes supplied features and model output; it is not a causal explanation.
- Model artifacts loaded through `joblib` must only come from trusted sources.

## Required Validation Before Any Field Study

At minimum: subgroup evaluation with consented data, calibration testing, drift monitoring, adverse-impact analysis, explanation fidelity checks, privacy impact assessment, accessibility review, and a documented human appeal process.
