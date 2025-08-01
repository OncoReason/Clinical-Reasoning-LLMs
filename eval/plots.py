"""
Script for visualizing evaluation metrics and model performance.

This script loads metrics from a pickle file and generates various plots:
- Macro‑F1 bar chart
- ROC curves
- Precision‑Recall curves
- Confusion matrices
- Predicted vs Actual Survival scatter
- Histogram, boxplot, and CDF of absolute errors
- Bar chart of generation metrics (BLEU, ROUGE, BERT‑F1)
- Scatter plot of generation quality vs classification F1

Usage:
    python plots.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import pickle

# Load metrics from a pickle file
with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

models   = list(metrics.keys())
versions = ["no_cot", "cot"]
colors   = {"med42":"C0", "meditron":"C1", "openbio":"C2", "ours":"C3", "ours_med42":"C4"}
styles   = {"no_cot":"-",    "cot":"--"}

plt.rcParams.update({'font.size': 20})

# 1) Bar chart of Macro‑F1
x    = np.arange(len(models))
w    = 0.35
f1_no = [metrics[m]["no_cot"]["macro_f1"] for m in models]
f1_co = [metrics[m]["cot"]["macro_f1"]   for m in models]
fig, ax = plt.subplots()
ax.bar(x - w/2, f1_no, w, label="no_CoT", color="gray")
ax.bar(x + w/2, f1_co, w, label="CoT",   color="black")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Macro‑F1")
ax.set_title("Macro‑F1 by Model (no_CoT vs CoT)")
ax.legend()
plt.tight_layout()
plt.show()

# 2) Combined ROC curves
fig, ax = plt.subplots()
for m in models:
    for v in versions:
        y_true  = metrics[m][v]["y_true_cls"]
        y_score = metrics[m][v].get("y_score", metrics[m][v]["y_pred_cls"])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        a          = auc(fpr, tpr)
        ax.plot(fpr, tpr, linestyle=styles[v], color=colors[m],
                label=f"{m}-{v} (AUC={a:.2f})")
ax.plot([0,1], [0,1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()

# 3) Combined Precision‑Recall curves
fig, ax = plt.subplots()
for m in models:
    for v in versions:
        y_true  = metrics[m][v]["y_true_cls"]
        y_score = metrics[m][v].get("y_score", metrics[m][v]["y_pred_cls"])
        p, r, _ = precision_recall_curve(y_true, y_score)
        ap      = auc(r, p)
        ax.plot(r, p, linestyle=styles[v], color=colors[m],
                label=f"{m}-{v} (AP={ap:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision‑Recall Curves")
ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()

# 4) Confusion matrices for best/worst by no_CoT Macro‑F1
best  = max(models, key=lambda m: metrics[m]["no_cot"]["macro_f1"])
worst= min(models, key=lambda m: metrics[m]["no_cot"]["macro_f1"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
for ax, m in zip([ax1, ax2], [best, worst]):
    cm = confusion_matrix(metrics[m]["no_cot"]["y_true_cls"],
                          metrics[m]["no_cot"]["y_pred_cls"])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    ax.set_title(f"{m} (no_CoT)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Living","Deceased"])
    ax.set_yticklabels(["Living","Deceased"])
fig.suptitle("Confusion Matrices")
plt.tight_layout()
plt.show()

# 5) Scatter Predicted vs Actual Survival (ours)
fig, ax = plt.subplots()
for v in versions:
    x = metrics["ours"][v]["y_true_reg"]
    y = metrics["ours"][v]["y_pred_reg"]
    ax.scatter(x, y, alpha=0.5, linestyle=styles[v],
               color=colors["ours"], label=f"ours-{v}")
ax.plot([min(x), max(x)], [min(x), max(x)], "k--")
ax.set_xlabel("Actual Survival (months)")
ax.set_ylabel("Predicted Survival (months)")
ax.set_title("ours: Predicted vs Actual Survival")
ax.legend()
plt.tight_layout()
plt.show()

# 6) Histogram of absolute errors
fig, ax = plt.subplots()
for m in models:
    for v in versions:
        errs = metrics[m][v]["abs_errors"]
        ax.hist(errs, bins=30, histtype="step",
                linestyle=styles[v], color=colors[m],
                label=f"{m}-{v}", density=True)
ax.set_xlabel("Absolute Error (months)")
ax.set_ylabel("Density")
ax.set_title("Error Distributions")
ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
# plt.tight_layout()
plt.show()

# 7) Boxplot of absolute errors
fig, ax = plt.subplots()
data   = [metrics[m][v]["abs_errors"] for m in models for v in versions]
labels = [f"{m}-{v}" for m in models for v in versions]
ax.boxplot(data, labels=labels, showfliers=False)
ax.set_ylabel("Absolute Error (months)")
ax.set_title("Error Distribution by Model/Version")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8) CDF of absolute errors
fig, ax = plt.subplots()
for m in models:
    for v in versions:
        errs = np.sort(metrics[m][v]["abs_errors"])
        cdf  = np.arange(1, len(errs)+1) / len(errs)
        ax.plot(errs, cdf, linestyle=styles[v], color=colors[m],
                label=f"{m}-{v}")
ax.set_xlabel("Absolute Error (months)")
ax.set_ylabel("CDF")
ax.set_title("CDF of Absolute Errors")
ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
# plt.tight_layout()
plt.show()

# 9) Bar chart of BLEU, ROUGE‑1, ROUGE‑2, BERT‑F1 (CoT only)
gen_metrics = ["bleu","rouge1","rouge2","bert_f1"]
x = np.arange(len(gen_metrics))
w = 0.2
fig, ax = plt.subplots()
for i, m in enumerate(models):
    vals = [metrics[m]["cot"][gm] for gm in gen_metrics]
    ax.bar(x + (i-1.5)*w, vals, width=w, color=colors[m], label=m)
ax.set_xticks(x)
ax.set_xticklabels(["BLEU","ROUGE‑1","ROUGE‑2","BERT‑F1"])
ax.set_title("Generation Quality (CoT)")
ax.legend()
plt.tight_layout()
plt.show()

# 10) Scatter Generation Quality vs Classification F1
fig, ax = plt.subplots()
for m in models:
    bleu = metrics[m]["cot"]["bleu"]
    f1   = metrics[m]["cot"]["macro_f1"]
    ax.scatter(bleu, f1, color=colors[m])
    ax.text(bleu, f1, m)
ax.set_xlabel("BLEU (CoT)")
ax.set_ylabel("Macro‑F1 (CoT)")
ax.set_title("Reasoning Quality vs Classification Performance")
plt.tight_layout()
plt.show()