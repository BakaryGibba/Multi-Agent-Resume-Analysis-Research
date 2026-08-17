# Reproducibility Guide

[Back to project overview](../README.md)

## What Can Be Reproduced

The repository supports three distinct checks:

1. Run the automated API tests without external services.
2. Re-run the notebook after obtaining the source dataset.
3. Import the sanitized n8n workflow and connect it to a local API and Groq account.

## Environment

Python 3.10 or newer is recommended. The serialized artifacts identify scikit-learn 1.6.1 and XGBoost 3.2.0, so those packages are pinned in `requirements.txt`. Bounded ranges are used for the remaining maintained dependencies.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pytest -q
```

## Re-Run The Experiment

1. Download the AI-Driven Resume Screening Dataset from its Kaggle source.
2. Open `ai-resume-merging.ipynb` in Kaggle or update its CSV path for your environment.
3. Run all cells from a clean kernel.
4. Compare the printed model table with the metrics in the README.
5. Treat newly generated `.pkl` files as version-specific artifacts.

The notebook uses `random_state=42` and a stratified 80/20 split. Hardware and dependency differences may produce small numerical variation.

## Run The Local API

```bash
python app.py
curl http://127.0.0.1:5000/health
```

Send all six required feature fields to `POST /predict`. The API rejects missing, negative, or non-numeric values. The model files are loaded relative to `app.py`, not the shell's current directory.

## Import The n8n Workflow

1. Import `Multi-Agent AI Resume Screening System (Groq).json` into n8n.
2. Configure `GROQ_API_KEY` in n8n's secret/environment mechanism, or replace the header expression with an n8n credential.
3. Start the Flask API on the same host as n8n, or update the prediction URL.
4. Use synthetic text in the test-data node.
5. Inspect Agent 1's JSON before allowing Agent 2 to run.

No API key, account identifier, execution history, or real resume is included in the export.

## Citation

Until archival proceedings or a journal version is confirmed, cite the work as accepted for presentation:

```bibtex
@unpublished{gibba2026multiagent,
  author = {Bakary Gibba and Sami Salama Hussen Hajjaj},
  title = {Multi-Agent Machine Learning Framework for Explainable Resume--Job Matching Using LLM-Based Agents},
  note = {Accepted for presentation at the 3rd International Symposium on Information Technology and Web Engineering (SITWE 2026)},
  year = {2026}
}
```

Update this record only after final publication metadata and a persistent identifier are issued.
