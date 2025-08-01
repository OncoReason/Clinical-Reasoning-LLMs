"""
Test script for SFT/CoT training and evaluation.

This script demonstrates loading configuration, preprocessing data, model setup,
training, evaluation, and inference for clinical reasoning LLMs.
"""

from datasets import load_dataset
import torch
import re
import os
import yaml
from trl import ModelConfig, TrlParser, ScriptArguments, SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import random
from pprint import pprint

# =========================
# Load YAML configuration
# =========================
with open('../sft_config.yaml', 'r') as file:
    config_dict = yaml.safe_load(file)

parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
script_args, training_args, model_args = parser.parse_dict(config_dict, allow_extra_keys=True)

# =========================
# Load datasets
# =========================
train_data = load_dataset('json', data_files='../msk_chord_cot_dataset_train_new.json', split='train') 
test_data = load_dataset('json', data_files='../msk_chord_cot_dataset_test_new.json', split='train') 

# =========================
# Model and Tokenizer Setup
# =========================
model_name = "meta-llama/Llama-3.1-8B-Instruct"
use_peft = True
use_bnb = True
quantization = "4bit"
hf_token = os.environ.get("HUGGINGFACE_TOKEN", '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')
cache_directory = "/scratch/rh3884/huggingface_models"

# =========================
# Prompt Templates (placeholders)
# =========================
# Note: The actual prompt templates will be made publicly available upon paper acceptance.
cot_prompt_template = """ """
prompt_template = """ """

# =========================
# Preprocessing Function
# =========================
def preprocess_data(examples):
    """
    Preprocess the dataset for training.

    Args:
        examples (dict): Batch of examples from the dataset.

    Returns:
        dict: Tokenized model inputs.
    """
    # Example using COT prompt template
    inputs = [
        cot_prompt_template.format(
            patient_data=examples["patient_data"][i],
            survival_status=examples["survival_status"][i],
            survival_months=examples["survival_months"][i],
            reasoning="\n".join(examples["chain_of_thought"][i]),
            comment=examples["comments"][i]
        )
        for i in range(len(examples["patient_data"]))
    ]

    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length"
    )
    return model_inputs

# =========================
# Tokenizer Initialization
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_directory)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# Map preprocessing to datasets
# =========================
train_data = train_data.map(preprocess_data, batched=True, batch_size=4)
eval_data = test_data.map(preprocess_data, batched=True, batch_size=4)

# =========================
# PEFT (LoRA) Configuration
# =========================
if use_peft:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],  # Adjust based on model architecture
        task_type="CAUSAL_LM"
    )
else:
    peft_config = None

# =========================
# BitsAndBytes Quantization Configuration
# =========================
if use_bnb:
    if quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
else:
    bnb_config = None

# =========================
# Model Initialization
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map="auto",
    use_auth_token=hf_token,
    cache_dir=cache_directory
)

# =========================
# Trainer Initialization
# =========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    peft_config=peft_config,
    processing_class=tokenizer,
    eval_dataset=eval_data,
)

# =========================
# Training
# =========================
trainer.train()

# =========================
# Evaluation
# =========================
results = trainer.evaluate()
print(results)

# =========================
# Inference Example
# =========================
infer_prompt_template = """\
Instruction: 
You are a technical hardware expert. For each hardware-related question provided, you must include a detailed chain-of-thought explanation that walks through your reasoning and analysis step by step, and then provide a clear, concise final answer. Your chain-of-thought should detail the technical considerations, specifications, compatibility checks, and any assumptions you make while evaluating the question.

Answer in the following format:
<reasoning>
[Your detailed step-by-step reasoning here]
</reasoning>
<answer>
Now provide a succinct final answer that directly responds to the user's question, drawing on your chain-of-thought analysis
</answer>

Question:
{problem}

#answer

"""

# Pick random samples from eval_data for inference
num_samples = 1  # Change this to however many you want
samples = random.sample(list(eval_data), num_samples)

print("\nRandom Predictions:\n")
for sample in samples:
    # Get input text depending on your dataset format
    input_text = infer_prompt_template.format(problem=sample["problem"])
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(trainer.model.device)
    
    # Generate model output
    with torch.no_grad():
        output_ids = trainer.model.generate(**inputs, max_new_tokens=1024)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Problem:\n{sample['problem']}\n")
    print(f"Model Output:\n{output_text}\n")
    print("=" * 50)

# Print ground truth reasoning and answer for the sample
print("<reasoning>\n" + sample["solution"] + "\n</reasoning>\n\n")
print("<answer>\n" + sample["answer"] + "\n</answer>")