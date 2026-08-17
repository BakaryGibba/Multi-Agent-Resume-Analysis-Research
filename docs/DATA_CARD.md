# Data Card

[Back to project overview](../README.md)

## Source

The experiments use the **AI-Driven Resume Screening Dataset** published on Kaggle by `sonalshinde123`. The notebook references:

`sonalshinde123/ai-driven-resume-screening-dataset/ai_resume_screening.csv`

The repository does not redistribute the dataset. Researchers must obtain it from the source and comply with the source's current terms and license.

## Observed Schema

The notebook records 30,000 rows and seven source columns:

| Field | Role |
| --- | --- |
| `years_experience` | Numeric input |
| `skills_match_score` | Numeric input |
| `education_level` | Categorical input |
| `project_count` | Numeric input |
| `resume_length` | Numeric input |
| `github_activity` | Numeric input |
| `shortlisted` | Binary prediction target |

No raw resumes, names, contact details, or protected-attribute fields are committed here.

## Transformations

- `shortlisted` is converted to a binary label.
- `education_level` is label encoded.
- Numeric missing values would be median imputed; categorical missing values would use the mode. The recorded run reports no missing values.
- Three deterministic composite features are added before training.
- The data is split 80/20 with stratification and `random_state=42`.

## Known Constraints

- The target reflects the dataset's label-generation process, not independently audited hiring outcomes.
- A tabular benchmark cannot represent the ambiguity and distribution shift of real resumes.
- Protected attributes are absent, so group fairness cannot be measured from this dataset alone.
- Proxy variables may still encode socioeconomic or educational bias.
- Dataset licensing and version metadata should be archived before formal replication.

## Appropriate Use

Use this dataset to study model comparison, pipeline design, and explanation interfaces. Do not treat it as evidence that a model is suitable for real employment decisions.
