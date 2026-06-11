# Evaluating Gemini Output Quality with the Vertex AI Gen AI Evaluation Service

A complete, runnable LLMOps example: score Gemini summaries with model-based
judges (summarization quality + groundedness), then run an A/B model migration
(`gemini-2.5-flash` vs `gemini-2.5-pro`) and apply a decision rule to gate the
upgrade.

## Files
- `gen_ai_evaluation.py` — the full runnable script (change only `PROJECT_ID`).
- `requirements.txt` — Python dependencies.

## Run it
```bash
pip install -r requirements.txt
gcloud auth application-default login
# edit gen_ai_evaluation.py and set PROJECT_ID to your project
python gen_ai_evaluation.py
```

## What it shows
1. Building a tiny evaluation dataset as a pandas DataFrame.
2. Scoring with `SUMMARIZATION_QUALITY` (1-5) and `GROUNDEDNESS` (0-1, the
   hallucination canary) via the Vertex AI Gen AI Evaluation Service.
3. Reading aggregate and per-row scores with the judge's plain-English reasoning.
4. Gating a model migration on evidence instead of vibes.

Companion article: "You Can't Ship What You Can't Measure: Why LLMOps Lives or
Dies on Evaluation (with Gemini on GCP)" — cognifydata.com
