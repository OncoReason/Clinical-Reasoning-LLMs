#!/usr/bin/env python

import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import torch

# === Config ===
MODEL_PATH = "data/Llama-3.1-8B-Instruct"
TEST_DATA_PATH = "msk_chord_cot_dataset_test_new.json"
OUTPUT_JSON_PATH = "model_predictions_new.json"
CACHE_DIR = "/scratch/huggingface_models"
BATCH_SIZE = 8
NUM_SAMPLES = 10

# === Prompt Template ===
# Note: The prompt templates will be made publicly available upon paper acceptance.
PROMPT_TEMPLATE = """ """ 

COT_PROMPT_TEMPLATE = """ """

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR
)
model.eval()

# === Load dataset ===
test_data = load_dataset('json', data_files=TEST_DATA_PATH, split='train')
# test_data = test_data.select(range(min(NUM_SAMPLES, len(test_data))))

# === Run batched inference ===
results = {}
device = model.device

for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    batch_dict = test_data[i:i + BATCH_SIZE]
    batch = [dict(zip(batch_dict.keys(), values)) for values in zip(*batch_dict.values())]

    batch_prompts = [
        COT_PROMPT_TEMPLATE.format(patient_data=item["patient_data"]) for item in batch
    ]
    patient_ids = [item["patient_id"] for item in batch]

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            # use_cache=True  # Enables KV caching for faster generation
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for pid, prompt, output in zip(patient_ids, batch_prompts, decoded_outputs):
        response_text = output.split("### Response:")[-1].strip()
        results[pid] = {
            "input": prompt,
            "model_response": response_text
        }

# === Save predictions ===
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Batched inference complete. Predictions saved to {OUTPUT_JSON_PATH}")
