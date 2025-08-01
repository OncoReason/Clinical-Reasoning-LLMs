import json
import os
import re
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for BERTScore: {device}")
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)

# === CONFIG ===
json_dir = "responses_no_sampling"
ground_truth_path = "../msk_chord_cot_dataset_test.json"

model_files = {
    "med42": {"no_cot": "med42_outputs.json", "cot": "med42_cot_outputs.json"},
    "meditron": {"no_cot": "meditron_outputs.json", "cot": "meditron_cot_outputs.json"},
    "openbio": {"no_cot": "openbio_outputs.json", "cot": "openbio_cot_outputs.json"},
    "ours": {"no_cot": "model_predictions.json", "cot": "model_predictions_cot.json"},
    "ours_med42": {"no_cot": "med42_model_predictions.json", "cot": "med42_model_predictions_cot.json"},
}

# === Label Normalization ===
def normalize_status(text):
    text = str(text).strip().upper()
    if "LIVING" in text or text == "0":
        return "0"
    if "DECEASED" in text or text == "1":
        return "1"
    return None

def extract_prediction(text):
    if not isinstance(text, str):
        return None, None
    text = re.sub(r'\s+', ' ', text)

    status_match = re.search(r"Overall Survival Status:\s*['\"]?(0:LIVING|1:DECEASED|0|1|LIVING|DECEASED)['\"]?", text, re.IGNORECASE)
    months_match = re.search(r"Estimated Overall Survival \(months\):\s*([0-9]+\.?[0-9]*)", text)

    raw_status = status_match.group(1) if status_match else None
    normalized_status = normalize_status(raw_status)
    months = float(months_match.group(1)) if months_match else None

    return normalized_status, months

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def clean_months(raw_month):
    if raw_month is None:
        return None
    try:
        return float(re.sub(r"[<>=]", "", str(raw_month)).strip())
    except ValueError:
        return None  # fallback if still unparseable

# === STEP 1: Load Ground Truth ===
dataset = load_dataset("json", data_files=ground_truth_path, split="train")


ground_truth = {
    row["patient_id"]: {
        "status": normalize_status(row["survival_status"]),
        "months": clean_months(row["survival_months"]),
        "chain_of_thought": "\n".join(row.get("chain_of_thought", [])),
        "comments": row.get("comments", ""),
        "survival_status": row.get("survival_status", ""),
        "survival_months": row.get("survival_months", ""),
    }
    for row in dataset
    if "patient_id" in row and "survival_status" in row and "survival_months" in row
}

# prepare rouge scorer once
rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

metrics = {}

response_format = """\
<reasoning>
{chain_of_thought}
</reasoning>

<comment>
{comments}
</comment>

<prediction>
Overall Survival Status: {survival_status}
Estimated Overall Survival (months): {survival_months}
</prediction>"
"""

# === STEP 2: Evaluate Models ===
for model_name, versions in model_files.items():
    metrics[model_name] = {}
    for version, file in versions.items():
        path = os.path.join(json_dir, file)
        predictions_raw = load_json(path)

        y_true_cls, y_pred_cls = [], []
        y_true_reg, y_pred_reg, abs_errors = [], [], []
        not_found_ids = []
        references, candidates = [], []

        for pid, gt in ground_truth.items():
            entry = predictions_raw.get(pid)
            if isinstance(entry, dict) and "model_response" in entry:
                text = entry["model_response"]
            else:
                text = entry

            text = text.split("### Response:")[-1].strip()

            # Golden response from deepseek R1
            golden_response = response_format.format(
                chain_of_thought=gt["chain_of_thought"],
                comments=gt["comments"],
                survival_status=gt["survival_status"],
                survival_months=gt["survival_months"],
            )

            # Collect for BLEU/ROUGE/BERTScore
            candidates.append(text)
            references.append(golden_response)

            pred_status, pred_months = extract_prediction(text)

            if pred_status is not None and gt["status"] is not None:
                y_true_cls.append(int(gt["status"]))
                y_pred_cls.append(int(pred_status))
            else:
                not_found_ids.append(pid)

            if pred_months is not None and gt["months"] is not None:
                y_true_reg.append(gt["months"])
                y_pred_reg.append(pred_months)
                abs_errors.append(abs(gt["months"] - pred_months))
            else:
                if pid not in not_found_ids:
                    not_found_ids.append(pid)
                

        print(f"\n📊 Results for {model_name} ({version})")

        # Classification
        print("\n🔹 Classification Metrics (Survival Status):")
        cr = classification_report(y_true_cls, y_pred_cls, output_dict=True)
        print(classification_report(y_true_cls, y_pred_cls, target_names=["LIVING", "DECEASED"]))

        # Regression
        if y_true_reg and y_pred_reg:
            mae = mean_absolute_error(y_true_reg, y_pred_reg)
            rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
            mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)
            r2 = r2_score(y_true_reg, y_pred_reg)
            print("\n🔸 Regression Metrics (Survival Months):")
            print(f"MAE  = {mae:.2f} months")
            print(f"RMSE = {rmse:.2f} months")
            print(f"MAPE = {mape:.2%}")
            print(f"R²   = {r2:.2f}")
        else:
            print("⚠️  No valid regression values found.")

        bleu, r1, r2_, rL, bert_f1 = None, None, None, None, None
        if version == "cot":
            # — Text Similarity Metrics —

            # BLEU (sacrebleu expects list of hypotheses and list-of-list of references)
            bleu = corpus_bleu(candidates, [[ref] for ref in references])
            print(f"\n📖 BLEU score: {bleu.score:.2f}")

            # ROUGE‑1/2/L
            scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
            rouge1, rouge2, rougeL = [], [], []
            for cand, ref in zip(candidates, references):
                scores = scorer.score(ref, cand)
                rouge1.append(scores["rouge1"].fmeasure)
                rouge2.append(scores["rouge2"].fmeasure)
                rougeL.append(scores["rougeL"].fmeasure)
            r1, r2_, rL = np.mean(rouge1), np.mean(rouge2), np.mean(rougeL)
            print(f"📖 ROUGE‑1 F1: {r1:.3f}")
            print(f"📖 ROUGE‑2 F1: {r2_:.3f}")
            print(f"📖 ROUGE‑L F1: {rL:.3f}")

            # BERTScore
            # P, R, F1 = bertscore_score(candidates, references, lang="en", rescale_with_baseline=True)
            P, R, F1 = bert_scorer.score(candidates, references)
            bert_f1 = F1.mean()
            print(f"📖 BERTScore  P: {P.mean():.3f}  R: {R.mean():.3f}  F1: {F1.mean():.3f}")

        metrics[model_name][version] = {
            "y_true_cls":    y_true_cls,
            "y_pred_cls":    y_pred_cls,
            "y_score":       [float(x) for x in y_pred_cls],  # fallback
            "y_true_reg":    y_true_reg,
            "y_pred_reg":    y_pred_reg,
            "abs_errors":    abs_errors,
            "accuracy":      cr["accuracy"],
            "macro_f1":      cr["macro avg"]["f1-score"],
            "mae":           mae,
            "rmse":          rmse,
            "r2":            r2,
            "bleu":          bleu,
            "rouge1":        r1,
            "rouge2":        r2_,
            "rougeL":        rL,
            "bert_f1":       bert_f1,
            "missing":       len(not_found_ids)
        }

        # Missing
        print(f"\n❓ Missing predictions: {len(not_found_ids)} → {not_found_ids[:5]}...\n")

# write out
with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)
print("Saved metrics.json")
