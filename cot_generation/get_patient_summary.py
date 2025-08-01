"""
Script to generate summarized patient clinical information from raw cBioPortal data.

This script processes patient JSON files, extracts and summarizes key clinical,
biomarker, treatment, and sample-specific information, and writes a summary per patient.

Usage:
    python get_patient_summary.py

Input:
    - Folder 'patient_data' containing one JSON file per patient.
    - 'attributes_metadata.json' describing attribute priorities and sources.

Output:
    - 'patient_summary.json' with summarized patient information.
"""

import json
import os
import statistics
from collections import defaultdict
from scipy.stats import linregress

# === Load attribute metadata ===
with open("attributes_metadata.json") as f:
    attr_meta = json.load(f)

# Build attribute priority and source dictionaries
attr_priority = {}
attr_source = {}
for attr in attr_meta:
    label = attr["displayName"].replace("(NLP)", "").strip()
    priority = int(attr["priority"])
    attr_priority[label] = (
        "HIGH" if priority >= 900 else "MEDIUM" if priority >= 500 else "LOW"
    )
    attr_source[label] = "Patient" if attr["patientAttribute"] else "Sample"

# === Define cancer-specific keywords and attribute mappings ===
cancer_specific_keywords = {
    "Breast Cancer": {"HER2", "ER", "PR", "HR", "PD-L1"},
    "Colorectal Cancer": {"MSI", "TMB", "Mutation Count"},
    "Non-Small Cell Lung Cancer": {"PD-L1", "EGFR", "ALK", "Smoking History (NLP)", "Smoking History"},
    "Pancreatic Cancer": {"MSI", "TMB"},
    "Prostate Cancer": {"Gleason", "PSA"},
}

# Map cancer types to relevant attributes for summary
cancer_type_attr_map = defaultdict(lambda: {"Patient": set(), "Sample": set()})
for attr, prio in attr_priority.items():
    for cancer, keywords in cancer_specific_keywords.items():
        if any(k in attr for k in keywords) or attr in {
            "Cancer Type", "Cancer Type Detailed", "Stage (Highest Recorded)",
            "Sex", "Current Age", "Overall Survival Status", "Smoking History", "Smoking History (NLP)"
        }:
            cancer_type_attr_map[cancer][attr_source[attr]].add(attr)

# Sample-level attributes to include by cancer type
sample_attrs_by_cancer = defaultdict(set)
sample_attrs_by_cancer.update({
    "Breast Cancer": {"Cancer Type Detailed", "Sample Type", "Metastatic Site", "Clinical Summary"},
    "Non-Small Cell Lung Cancer": {"Cancer Type Detailed", "Sample Type", "Metastatic Site", "MSI Type", "MSI Score", "Clinical Group"},
    "Colorectal Cancer": {"Clinical Summary", "Diagnosis Description", "Cancer Type Detailed", "Sample Type", "Clinical Group", "Metastatic Site", "MSI Type", "MSI Score", "Primary Tumor Site"},
    "Prostate Cancer": {"Clinical Summary", "Cancer Type Detailed", "Sample Type", "Clinical Group", "Metastatic Site", "Gleason Score Reported on Sample"},
    "Pancreatic Cancer": {"Clinical Summary", "Cancer Type Detailed", "Sample Type", "Metastatic Site", "MSI Type", "MSI Score", "Clinical Group"}
})

# === Summarize lab tests ===
def summarize_lab_tests(lab_tests, cancer_types):
    """
    Summarize key tumor marker lab tests for a patient.

    Args:
        lab_tests (list): List of lab test records.
        cancer_types (set): Set of cancer types for the patient.

    Returns:
        str: Summary string of tumor marker trends and values.
    """
    def compute_stats(values):
        """
        Compute statistics for a list of (day, value) tuples.
        """
        if len(values) < 2:
            return {"avg": round(values[0][1], 1), "note": "single value"}
        values.sort()
        days, results = zip(*values)
        if len(set(days)) == 1:
            return {
                "avg": round(statistics.mean(results), 1),
                "note": "multiple results from same day"
            }
        slope, *_ = linregress(days, results)
        trend = "rising" if slope > 0 else "falling" if slope < 0 else "stable"
        return {
            "avg": round(statistics.mean(results), 1),
            "trend": trend,
            "slope": round(slope, 3),
            "start": results[0],
            "end": results[-1],
            "peak": max(results),
            "days": days[-1] - days[0]
        }

    def extract_values(aliases):
        """
        Extract (day, value) tuples for given marker aliases.
        """
        return [
            (int(entry.get("DAYS_FROM_DIAGNOSIS", entry.get("START_DATE", 0))), float(entry["RESULT"]))
            for entry in lab_tests
            if entry.get("TEST", "").strip().upper() in aliases
            and entry.get("RESULT", "").replace('.', '', 1).isdigit()
        ]

    markers = []
    tests = {
        "CEA": {"alias": {"CEA"}, "cancers": {"Colorectal Cancer", "Pancreatic Cancer", "Non-Small Cell Lung Cancer", "Breast Cancer"}},
        "CA15-3": {"alias": {"CA_15-3", "CA15-3", "CA 15-3"}, "cancers": {"Breast Cancer"}},
        "CA19-9": {"alias": {"CA_19-9", "CA19-9"}, "cancers": {"Colorectal Cancer", "Pancreatic Cancer"}},
        "PSA": {"alias": {"PSA"}, "cancers": {"Prostate Cancer"}}
    }

    for marker, config in tests.items():
        if not cancer_types.intersection(config["cancers"]):
            continue
        values = extract_values(config["alias"])
        if values:
            stats = compute_stats(values)
            if "start" in stats:
                markers.append(
                    f"- {marker}: {stats['start']} → {stats['end']} over {stats['days']} days ({stats['trend']}); "
                    f"avg={stats['avg']}; peak={stats['peak']}"
                )
            elif "note" in stats:
                markers.append(f"- {marker}: {stats['avg']} ({stats['note']})")
            else:
                markers.append(f"- {marker}: {stats['avg']}")

    return "Key Tumor Markers:\n" + "\n".join(markers) if markers else ""

# === Treatment summarization ===
def extract_treatment_summary(events, type_key, label):
    """
    Summarize treatments of a given type for a patient.

    Args:
        events (list): List of treatment event dicts.
        type_key (str): Subtype to filter (e.g., 'chemo').
        label (str): Label for the summary (e.g., 'Chemotherapy').

    Returns:
        str: Summary string for the treatment type.
    """
    lines, dates = [], []
    for t in events:
        if t.get("SUBTYPE", "").lower() == type_key:
            agent = t.get("AGENT", "").replace("(NLP)", "").strip()
            start, stop = t.get("START_DATE"), t.get("STOP_DATE")
            dates.append((start, stop))
            lines.append(agent.upper() if agent else "[Unknown Agent]")
    if not lines:
        return ""
    agents = ", ".join(sorted(set(lines)))
    all_dates = [int(d) for pair in dates for d in pair if d and str(d).isdigit()]
    date_range = f"Days: {min(all_dates)}–{max(all_dates)}" if all_dates else "Days: Unknown"
    return f"- {label}: {agents}, {date_range}"

# === Main patient summary generator ===
def extract_patient_info(record):
    """
    Extract and summarize clinical, biomarker, treatment, and sample-specific info for a patient.

    Args:
        record (dict): Raw patient record.

    Returns:
        dict: Summarized patient information.
    """
    clinical = record.get("CLINICAL_DATA", [])
    patient_id = record.get("patient_id", [])
    survival_status = next((i["Value"] for i in clinical if i["Attribute"] == "Overall Survival Status"), "N/A")
    survival_months = next((i["Value"] for i in clinical if i["Attribute"] == "Overall Survival (Months)"), "N/A")

    cancer_types = set()
    sample_data = defaultdict(dict)
    for item in clinical:
        attr = item.get("Attribute", "").replace("(NLP)", "").strip()
        for k, v in item.items():
            if k.startswith("P-") and "-T" in k and v and v.lower() != "n/a":
                sample_data[k][attr] = v
                if attr == "Cancer Type":
                    cancer_types.add(v)

    clinical_attrs, tumor_sites, biomarkers = [], set(), []
    for item in clinical:
        label = item.get("Attribute", "").replace("(NLP)", "").strip()
        value = item.get("Value", "").replace("(NLP)", "").strip()
        if not value or value.lower() == "n/a" or label in {
            "Overall Survival Status", "Overall Survival (Months)",
            "Number of Tumor Registry Entries", "Number of Samples Per Patient",
            "Race", "Ethnicity"
        }:
            continue
        # Biomarker and attribute logic
        if label in {"HER2", "PD-L1", "HR", "ER", "PR"}:
            if label == "PD-L1" or ("Breast Cancer" in cancer_types and label != "PD-L1"):
                biomarkers.append(f"{label}={value}")
        elif "Tumor Site:" in label and value.lower() in {"yes", "true"}:
            tumor_sites.add(label.split("Tumor Site:")[-1].strip())
        elif label == "Smoking History":
            if "Non-Small Cell Lung Cancer" in cancer_types:
                clinical_attrs.append(f"Smoking History={value}")
        elif label == "Stage (Highest Recorded)":
            clinical_attrs.append(f"Cancer {value}")
        elif label == "Sex":
            if {"Non-Small Cell Lung Cancer", "Colorectal Cancer", "Pancreatic Cancer"}.intersection(cancer_types):
                clinical_attrs.append(f"{label}={value}")
        else:
            src = attr_source.get(label, "Patient")
            if label in cancer_type_attr_map[next(iter(cancer_types), "")][src]:
                clinical_attrs.append(f"{label}={value}")

    blocks = []
    if clinical_attrs:
        blocks.append("Clinical Attributes: " + "; ".join(sorted(clinical_attrs)))
    if biomarkers:
        blocks.append("Biomarkers: " + "; ".join(sorted(biomarkers)))
    if tumor_sites:
        blocks.append("Tumor Sites: " + ", ".join(sorted(tumor_sites)))

    # Treatment summaries
    treatments = record.get("Treatment", [])
    extra_therapy = record.get("TREATMENT", [])
    all_radiation = [t for t in (treatments + extra_therapy) if "Radiation" in t.get("SUBTYPE", "")]

    radiation_dates = [
        int(t.get("START_DATE"))
        for t in all_radiation
        if t.get("START_DATE") not in [None, "", "N/A"] and str(t.get("START_DATE")).lstrip('-').isdigit()
    ]
    radiation_summary = ""
    if radiation_dates:
        sorted_dates = sorted(radiation_dates)
        radiation_summary = f"- Radiation Therapy: Days {', '.join(str(d) for d in sorted_dates)}"

    treatment_lines = [
        extract_treatment_summary(treatments, "chemo", "Chemotherapy"),
        extract_treatment_summary(treatments, "immuno", "Immunotherapy"),
        extract_treatment_summary(treatments, "investigational", "Investigational Treatments"),
        extract_treatment_summary(treatments, "targeted", "Targeted Treatments")
    ]
    if radiation_summary:
        treatment_lines.append(radiation_summary)
    blocks.append("Treatments:\n" + "\n".join([line for line in treatment_lines if line]))

    # Lab test summary
    if (lab := summarize_lab_tests(record.get("LAB_TEST", []), cancer_types)):
        blocks.append(lab)

    # Multiple diagnoses
    if len(cancer_types) > 1:
        blocks.append("Diagnoses: " + ", ".join(sorted(cancer_types)))

    # Sample-specific information
    if sample_data:
        cancer_type_to_samples = defaultdict(list)
        for sid, attrs in sample_data.items():
            ct = attrs.get("Cancer Type", "Unknown Cancer")
            cancer_type_to_samples[ct].append((sid, attrs))

        for ct, samples in cancer_type_to_samples.items():
            for sid, attrs in samples:
                blocks.append(f"Sample-Specific Information ({ct}):")
                selected = {}
                for k, v in attrs.items():
                    if k == "ICD-O Histology Description":
                        if ct in {"Colorectal Cancer", "Pancreatic Cancer", "Prostate Cancer"}:
                            if v not in {"Adenocarcinoma, Nos", "Carcinoma, Nos"}:
                                selected[k] = v
                    elif k in sample_attrs_by_cancer[ct]:
                        selected[k] = v
                for k, v in selected.items():
                    blocks.append(f"- {k}: {v}")

    return {
        "patient_data": "\n".join(blocks),
        "survival_status": survival_status,
        "survival_months": survival_months,
        "cancer_type": sorted(cancer_types),
        "patient_id": patient_id
    }

# === Main execution ===
if __name__ == "__main__":
    input_folder = "patient_data"
    output_file = "patient_summary.json"
    all_results = []

    # Process each patient JSON file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path) as f:
                record = json.load(f)
            try:
                result = extract_patient_info(record)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Sort results by patient_id and write to output file
    all_results.sort(key=lambda x: x.get("patient_id", ""))
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)