"""
Benchmarking script for clinical reasoning LLMs using patient summary data.

This script loads a test dataset, prepares prompts (basic and chain-of-thought),
runs inference with various Hugging Face models, and saves the generated outputs.
It supports both pipeline and standard model inference, as well as batched generation.

Note: Prompt templates are placeholders and will be made available upon paper acceptance.
"""

import os
import json
from tqdm import tqdm
from datasets import load_dataset

from torch.utils.data import DataLoader
from transformers import default_data_collator
import transformers

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# Dataset Loading
# =========================

# Load the test dataset from JSON file
test_data = load_dataset('json', data_files='msk_chord_cot_dataset_test.json', split='train')

# =========================
# Prompt Templates
# =========================

# Note: The prompt templates will be made publicly available upon paper acceptance.
prompt_template = """ """
cot_prompt_template = """ """

# =========================
# Prompt Preparation
# =========================

def prepare_prompts(data):
    """
    Prepare basic and chain-of-thought (COT) prompts for each patient record.

    Args:
        data (iterable): Dataset containing patient records.

    Returns:
        tuple: (prompts_basic, prompts_cot), each a list of dicts with patient_id and prompt text.
    """
    prompts_basic = []
    prompts_cot = []
    for ex in data:
        pid = ex["patient_id"]
        pdata = ex["patient_data"]

        base_prompt = prompt_template.format(patient_data=pdata)
        cot_prompt = cot_prompt_template.format(patient_data=pdata)

        prompts_basic.append({"patient_id": pid, "text": base_prompt})
        prompts_cot.append({"patient_id": pid, "text": cot_prompt})
    return prompts_basic, prompts_cot

# =========================
# Inference Functions
# =========================

def generate_responses(model, tokenizer, prompts, device="cuda", **gen_kwargs):
    """
    Generate model responses for a list of prompts (single example at a time).

    Args:
        model: Hugging Face model.
        tokenizer: Corresponding tokenizer.
        prompts (list): List of dicts with 'patient_id' and 'text'.
        device (str): Device to run inference on.
        gen_kwargs: Generation keyword arguments.

    Returns:
        dict: Mapping patient_id to generated response.
    """
    results = {}
    for item in tqdm(prompts):
        # Tokenize with attention mask
        encoded = tokenizer(item["text"], return_tensors='pt', truncation=True, padding=True)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        # Set pad_token_id if not defined
        if "pad_token_id" not in gen_kwargs:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        output_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results[item["patient_id"]] = prediction

    return results

def generate_responses_batched(model, tokenizer, prompts, batch_size=8, device="cuda", **gen_kwargs):
    """
    Generate model responses for a list of prompts in batches.

    Args:
        model: Hugging Face model.
        tokenizer: Corresponding tokenizer.
        prompts (list): List of dicts with 'patient_id' and 'text'.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
        gen_kwargs: Generation keyword arguments.

    Returns:
        dict: Mapping patient_id to generated response.
    """
    results = {}

    # Prepare a list of texts and patient_ids
    texts = [item["text"] for item in prompts]
    patient_ids = [item["patient_id"] for item in prompts]

    # Tokenize all prompts
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Build dataset as list of dictionaries
    dataset = [
        {
            "patient_id": pid,
            "input_ids": input_id,
            "attention_mask": attn_mask,
        }
        for pid, input_id, attn_mask in zip(patient_ids, encodings["input_ids"], encodings["attention_mask"])
    ]

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)

    # Set pad_token_id explicitly
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    model.to(device)
    model.eval()

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # patient_ids are not batched by default_data_collator, keep track separately
        batch_size_actual = input_ids.shape[0]
        pids = patient_ids[:batch_size_actual]
        patient_ids = patient_ids[batch_size_actual:]  # Update the list to remove used IDs

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for pid, pred in zip(pids, decoded):
            results[pid] = pred

    return results

# =========================
# Model Runner
# =========================

def run_model(model_name, out_prefix, tokenizer_cls=AutoTokenizer, model_cls=AutoModelForCausalLM, chat_template=False, pipe_mode=False):
    """
    Load a model and tokenizer, prepare prompts, run inference, and save outputs.

    Args:
        model_name (str): Hugging Face model name or path.
        out_prefix (str): Prefix for output JSON files.
        tokenizer_cls: Tokenizer class to use.
        model_cls: Model class to use.
        chat_template (bool): Whether to use a chat template for prompts.
        pipe_mode (bool): Whether to use Hugging Face pipeline for inference.
    """
    print(f"[{out_prefix}] Loading model: {model_name}")
    
    if pipe_mode:
        # Use Hugging Face pipeline for text generation (single GPU)
        pipe = pipeline("text-generation", model=model_name, device=0, model_kwargs={"torch_dtype": torch.bfloat16})

    else:
        # Standard model/tokenizer loading
        if model_name == 'chaoyi-wu/PMC_LLAMA_7B':
            tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
            tokenizer.padding_side = "left"
            model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B', device_map="auto", torch_dtype=torch.float16)
        else:
            tokenizer = tokenizer_cls.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token  # Set a pad token
            tokenizer.padding_side = "left"           # Decoder-only models need left padding
            tokenizer.model_max_length = 2048         # Or whatever fits your model size

            model = model_cls.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    prompts_basic, prompts_cot = prepare_prompts(test_data)

    if pipe_mode:
        results_basic = {}
        results_cot = {}

        # Only COT prompts are run here, but you can add basic prompts if needed
        for prompt_set, name, result_dict in [(prompts_cot, 'cot', results_cot)]:    
            for ex in tqdm(prompt_set):
                prompt = ex["text"]

                # Optionally apply chat template
                if chat_template:
                    prompt = (
                        "[INST] <<SYS>> You are an expert medical assistant from Saama AI Labs. <</SYS>>\n"
                        f"[INST] {prompt} [/INST]"
                    )
                    terminators = [pipe.tokenizer.eos_token_id]
                    output = pipe(prompt, max_new_tokens=10240, temperature=0.7, do_sample=False, top_p=0.9, eos_token_id=terminators)
                else:
                    terminators = [pipe.tokenizer.eos_token_id]
                    output = pipe(prompt, max_new_tokens=10240, temperature=0.7, do_sample=False, top_p=0.9)
                # Remove prompt from output to get only the generated text
                result_dict[ex["patient_id"]] = output[0]["generated_text"][len(prompt):]

        # Save results to JSON
        json.dump(results_cot, open(f"{out_prefix}_cot_outputs.json", "w"), indent=2)

    else:
        # Batched generation for COT prompts (can add basic if needed)
        results_cot = generate_responses_batched(model, tokenizer, prompts_cot, max_new_tokens=10240, temperature=0.7, do_sample=False)

        # Save results to JSON
        json.dump(results_cot, open(f"{out_prefix}_cot_outputs.json", "w"), indent=2)

# =========================
# Model Execution Section
# =========================

if __name__ == "__main__":
    # Run Med42 model (Llama3-Med42-8B)
    run_model("m42-health/Llama3-Med42-8B", "med42") 

    # Run Meditron model 
    run_model("epfl-llm/meditron-7b", "meditron")

    # Run OpenBioLLM-Llama3-8B in pipeline mode (chat_template can be toggled)
    run_model("aaditya/OpenBioLLM-Llama3-8B", "openbio", chat_template=False, pipe_mode=True)
    # run_model("aaditya/OpenBioLLM-Llama3-8B", "openbio", chat_template=True, pipe_mode=True)

    # Example for PMC_LLAMA_7B 
    # run_model("chaoyi-wu/PMC_LLAMA_7B", "pmc_llama")


  