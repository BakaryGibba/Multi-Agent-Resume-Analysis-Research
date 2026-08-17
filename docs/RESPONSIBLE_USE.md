# Responsible Use

[Back to project overview](../README.md)

## Safety Position

Employment is a high-impact domain. This prototype must remain advisory and experimental. A probability or generated explanation must never become the sole basis for an employment decision.

## Core Risks

| Risk | Why it matters | Required control |
| --- | --- | --- |
| Historical or proxy bias | Apparently neutral fields can reproduce unequal outcomes | Representative evaluation and subgroup audits |
| LLM extraction error | Unstructured text may be misread or fabricated | Schema validation and human verification |
| Explanation overconfidence | Fluent text can appear more faithful than it is | Display source features and model output beside the narrative |
| Privacy loss | Resumes contain sensitive personal data | Data minimization, consent, retention limits, and access controls |
| Automation bias | Reviewers may defer to a score | Independent review, override, and appeal mechanisms |
| Distribution shift | Real applicants differ from benchmark data | External validation and continuous monitoring |

## Human Oversight

A reviewer should be able to inspect input provenance, correct extraction errors, ignore the model recommendation, document a decision independently, and offer an accessible appeal path. The system should not hide uncertainty behind a binary label.

## Data Handling

- Do not commit resumes, API keys, n8n credentials, or identifiable candidate data.
- Use synthetic or explicitly consented test data.
- Separate experiment logs from personal information.
- Define retention and deletion procedures before any field study.
- Treat serialized model files as executable artifacts and load only trusted copies.

## Explanation Boundary

Agent 3 produces a natural-language summary of inputs and output. It does not expose the model's internal causal reasoning and must not be described as proof that a decision is fair or correct.
