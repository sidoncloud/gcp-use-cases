# =============================================================================
# Measuring Gemini Output Quality with the Vertex AI Gen AI Evaluation Service
# =============================================================================
# A complete, runnable LLM evaluation harness. Change PROJECT_ID below, then run.
#
# Setup (once):
#   pip install --upgrade "google-cloud-aiplatform[evaluation]"
#   gcloud auth application-default login
#
# Companion article: "You Can't Ship What You Can't Measure: Why LLMOps Lives
# or Dies on Evaluation (with Gemini on GCP)" by Cognify Data.
# =============================================================================

import pandas as pd

import vertexai
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
from google import genai

PROJECT_ID = "your-project-id"   # change this to your Google Cloud project
LOCATION = "us-central1"

# Eval runs execute and bill as batch prediction jobs under the hood.
vertexai.init(project=PROJECT_ID, location=LOCATION)

# For generating candidate responses we use the current Gen AI SDK client.
genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# -----------------------------------------------------------------------------
# 1. Build a tiny evaluation dataset
# -----------------------------------------------------------------------------
eval_rows = [
    {
        "prompt": (
            "Summarize for an executive: Q3 revenue rose 12 percent to $4.2M, "
            "driven by enterprise renewals. Churn held flat at 2 percent. "
            "Support costs grew 8 percent after we hired three agents."
        ),
        "response": (
            "Q3 revenue grew 12 percent to $4.2M on strong enterprise renewals, "
            "with churn steady at 2 percent and support costs up 8 percent."
        ),
    },
    {
        "prompt": (
            "Summarize for an executive: A datacenter outage on Tuesday took the "
            "EU region offline for 47 minutes. Root cause was a failed network "
            "switch. No data was lost and failover is now automated."
        ),
        "response": (
            "A 47-minute EU outage on Tuesday was caused by a failed switch. No "
            "data was lost, and failover has since been automated."
        ),
    },
    {
        "prompt": (
            "Summarize for an executive: The mobile app shipped dark mode and "
            "offline sync this month. Crash rate dropped 30 percent. App store "
            "rating climbed from 4.1 to 4.5 stars."
        ),
        # A deliberately weak summary so the metrics have something to penalize.
        "response": "The app got some updates and people seem happier about it.",
    },
]

eval_dataset = pd.DataFrame(eval_rows)
print(f"Built eval dataset with {len(eval_dataset)} examples")


# -----------------------------------------------------------------------------
# 2. Define metrics and run the evaluation
# -----------------------------------------------------------------------------
metrics = [
    MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY,
    MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
]

summarization_eval = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment="summary-quality-eval",
)

result = summarization_eval.evaluate()
print("Evaluation complete.")


# -----------------------------------------------------------------------------
# 3. Read the scores, and the judge's reasoning
# -----------------------------------------------------------------------------
print("Aggregate scores across all rows")
for metric_name in ["summarization_quality", "groundedness"]:
    mean = result.summary_metrics.get(f"{metric_name}/mean")
    std = result.summary_metrics.get(f"{metric_name}/std")
    print(f"  {metric_name:25s} mean={mean:.2f}  std={std:.2f}")

for i, row in result.metrics_table.iterrows():
    print(f"Row {i}")
    print(f"  summarization_quality  {row['summarization_quality/score']:.1f} / 5")
    print(f"  groundedness           {row['groundedness/score']:.1f} / 1")
    print(f"  why: {row['summarization_quality/explanation']}")


# -----------------------------------------------------------------------------
# 4. Gate a model migration: gemini-2.5-flash vs gemini-2.5-pro
# -----------------------------------------------------------------------------
migration_dataset = pd.DataFrame(
    {"prompt": [row["prompt"] for row in eval_rows]}
)


def make_responder(model_name: str):
    """Return a callable that takes a prompt and returns a response string."""
    def respond(prompt: str) -> str:
        return genai_client.models.generate_content(
            model=model_name, contents=prompt
        ).text
    return respond


candidates = {
    "gemini-2.5-flash": make_responder("gemini-2.5-flash"),
    "gemini-2.5-pro": make_responder("gemini-2.5-pro"),
}

comparison = {}
for name, responder in candidates.items():
    task = EvalTask(
        dataset=migration_dataset,
        metrics=metrics,
        experiment="model-migration-eval",
    )
    # Passing model= makes the service call the model to produce responses,
    # then score them. This is what makes the comparison fair.
    candidate_result = task.evaluate(model=responder)
    comparison[name] = candidate_result.summary_metrics

print(f"Scored {len(comparison)} candidate models on the same prompts.")


# -----------------------------------------------------------------------------
# 5. Compare side by side and apply the decision rule
# -----------------------------------------------------------------------------
rows = []
for name, summary in comparison.items():
    rows.append({
        "model": name,
        "quality_mean": summary.get("summarization_quality/mean"),
        "groundedness_mean": summary.get("groundedness/mean"),
    })
compare_df = pd.DataFrame(rows).set_index("model")

print("Model comparison (higher is better on both means)")
print(compare_df.round(3).to_string())

flash = compare_df.loc["gemini-2.5-flash"]
pro = compare_df.loc["gemini-2.5-pro"]
quality_delta = pro["quality_mean"] - flash["quality_mean"]
ground_delta = pro["groundedness_mean"] - flash["groundedness_mean"]

print(f"quality delta (pro - flash):      {quality_delta:+.2f}")
print(f"groundedness delta (pro - flash): {ground_delta:+.2f}")

# Decision rule: prefer the cheaper model unless pro wins clearly on quality;
# any groundedness regression is an automatic disqualification.
if ground_delta < 0:
    print("-> stay on gemini-2.5-flash: pro regressed on groundedness.")
elif quality_delta > 0.5 and ground_delta >= 0:
    print("-> promote gemini-2.5-pro: clear quality win, no honesty cost.")
else:
    print("-> stay on gemini-2.5-flash: gaps are within noise, cheaper wins.")
