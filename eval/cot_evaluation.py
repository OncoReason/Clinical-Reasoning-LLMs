import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === Config ===
PATIENT_SUMMARY_FILE = "patient_data.json"     # List of dicts with 'patient_id' and 'patient_data'
COT_FILE = "cot.json"
PROMPT_FILE = "prompt.txt"                     

# === Load Sentence-BERT model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load Data ===
with open(PATIENT_SUMMARY_FILE) as f:
    summaries_raw = json.load(f)

with open(COT_FILE) as f:
    cot_data = json.load(f)

# Map patient_id → summary
summary_data = {
    item["patient_id"]: item["patient_data"]
    for item in summaries_raw
    if "patient_id" in item and "patient_data" in item
}

# Load prompt if needed
prompt = None
if PROMPT_FILE:
    with open(PROMPT_FILE) as f:
        prompt = f.read()

# === Define Evaluation Function ===
def evaluate_cot(summary: str, cot_steps: list, prompt: str = None):
    summary_emb = model.encode(summary, convert_to_tensor=True)
    cot_embs = model.encode(cot_steps, convert_to_tensor=True)

    relevance_scores = [util.cos_sim(summary_emb, step_emb).item() for step_emb in cot_embs]
    coherence_scores = [
        util.cos_sim(cot_embs[i], cot_embs[i+1]).item()
        for i in range(len(cot_embs) - 1)
    ] if len(cot_embs) > 1 else []

    prompt_scores = []
    if prompt:
        prompt_emb = model.encode(prompt, convert_to_tensor=True)
        prompt_scores = [util.cos_sim(prompt_emb, step_emb).item() for step_emb in cot_embs]

    return {
        "avg_relevance": round(sum(relevance_scores)/len(relevance_scores), 4),
        "min_relevance": round(min(relevance_scores), 4),
        "avg_coherence": round(sum(coherence_scores)/len(coherence_scores), 4) if coherence_scores else None,
        "max_prompt_overlap": round(max(prompt_scores), 4) if prompt_scores else None,
        "num_steps": len(cot_steps),
    }

# === Evaluate all patient CoTs ===
results = {}
for pid, cot_json_string in cot_data.items():
    if pid not in summary_data:
        continue
    try:
        cot_dict = json.loads(cot_json_string)
        cot_steps = cot_dict.get("chain_of_thought", [])
        if not cot_steps:
            continue
        summary = summary_data[pid]
        metrics = evaluate_cot(summary, cot_steps, prompt)
        results[pid] = metrics
    except Exception as e:
        print(f"Skipping {pid} due to error: {e}")

# === Save and Print Summary ===
df = pd.DataFrame.from_dict(results, orient="index")
df.index.name = "patient_id"
df.to_csv("cot_eval_results.csv")
print("Saved evaluation results to cot_eval_results.csv")
print(df.head())
