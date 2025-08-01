import os
import requests
import json
from tqdm import tqdm

BASE_URL = "https://www.cbioportal.org/api"
HEADERS = {"Accept": "application/json"}
STUDY_ID = "msk_chord_2024"
OUTPUT_DIR = "patient_data_api"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Get all patients
patients_url = f"{BASE_URL}/studies/{STUDY_ID}/patients"
patients = requests.get(patients_url, headers=HEADERS).json()

# Step 2: Get all samples
samples_url = f"{BASE_URL}/studies/{STUDY_ID}/samples"
samples = requests.get(samples_url, headers=HEADERS).json()

# Index samples by patientId
samples_by_patient = {}
for sample in samples:
    pid = sample["patientId"]
    samples_by_patient.setdefault(pid, []).append(sample)

# Step 3: Process each patient
for patient in tqdm(patients, desc="Downloading full data per patient"):
    patient_id = patient["patientId"]
    output_path = os.path.join(OUTPUT_DIR, f"{patient_id}.json")

    # ✅ Skip if file already exists
    if os.path.exists(output_path):
        continue

    # Get patient-level clinical data
    clinical_url = f"{BASE_URL}/studies/{STUDY_ID}/patients/{patient_id}/clinical-data"
    clinical_data = requests.get(clinical_url, headers=HEADERS).json()

    # Collect samples + their clinical data
    patient_samples = []
    for sample in samples_by_patient.get(patient_id, []):
        sample_id = sample["sampleId"]
        
        sample_clinical_url = f"{BASE_URL}/studies/{STUDY_ID}/samples/{sample_id}/clinical-data"
        sample_clinical_data = requests.get(sample_clinical_url, headers=HEADERS).json()

        patient_samples.append({
            "sample_info": sample,
            "sample_clinical_data": sample_clinical_data
        })
    print("processed patient :"+patient_id)
    # Save combined data
    with open(output_path, "w") as f:
        json.dump({
            "patient_id": patient_id,
            "patient_clinical_data": clinical_data,
            "samples": patient_samples
        }, f, indent=4)
