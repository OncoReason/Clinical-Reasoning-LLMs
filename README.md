# Clinical-Reasoning-LLMs

A toolkit for training, inference, and evaluation of large language models (LLMs) on clinical reasoning tasks using patient data and chain-of-thought (CoT) approaches.

---

## Steps Overview

1. **Create Chain-of-Thought (CoT) Dataset**
   - Use `cot_generation/data_fetch.py` to fetch and aggregate patient/sample data from cBioPortal.
   - Use `cot_generation/get_patient_summary.py` to summarize patient data for modeling.
   - Use `cot_generation/get_cot.py` to generate CoT reasoning for each patient (requires DeepSeek API).

2. **Train SFT Model**
   - Prepare configuration in `sft_config.yaml`.
   - Run SFT training using either:
     ```sh
     bash train_sft.sh
     ```
     or
     ```sh
     python src/train_sft.py --config sft_config.yaml
     ```
   - For CoT SFT, use:
     ```sh
     cd reasoning_models
     accelerate launch --config_file src/zero2.yaml src/train_sft_cot.py --config sft_config.yaml
     ```

3. **Train GRPO Model**
   - Use `src/train_grpo.py` for GRPO-specific training with custom reward functions.

4. **Inference**
   - Use `infer.py` to run inference on new patient data with trained models.
   - Outputs include model predictions and chain-of-thought reasoning.

5. **Evaluation Metrics**
   - Use `eval/evaluate.py` for general evaluation (classification, regression).
   - Use `eval/cot_evaluation.py` for CoT-specific evaluation using embedding-based metrics.
   - Visualize results with `eval/plot_metrics.py`, `eval/plots.py`, or `eval/plots.ipynb`.

---

## Project Structure

```
benchmarking.py                # Benchmarking models and configurations
infer.py                       # Inference script
sft_config.yaml                # SFT training configuration
train_sft.sh                   # SFT training shell script

cot_generation/
    data_fetch.py              # Fetch patient/sample data from cBioPortal
    get_cot.py                 # Generate CoT reasoning using DeepSeek API
    get_patient_summary.py     # Summarize patient data

eval/
    cot_evaluation.py          # CoT evaluation (embedding-based metrics)
    evaluate.py                # General evaluation
    plot_metrics.py            # Plotting metrics
    plots.py                   # Visualization scripts
    plots.ipynb                # Visualization notebook

src/
    base_sft_trainer.py        # Base SFT trainer (PEFT/BNB support)
    base_trainer.py            # Base GRPO trainer
    commands.txt               # Example CLI commands
    common.py                  # Shared utilities
    monitor.py                 # Training monitor dashboard
    test.py                    # SFT/CoT training and inference example
    train_grpo.py              # GRPO training script
    train_sft_cot.py           # SFT with CoT training script
    train_sft.py               # SFT training script
    zero2.yaml                 # DeepSpeed config
```

---

## Configuration

- Training and evaluation parameters are set in YAML files (e.g., `sft_config.yaml`, `src/zero2.yaml`).
- Example CLI commands are in `src/commands.txt`.

---

## Requirements

See `requirements.txt` for all dependencies.

---

## License

*Specify your license here (e.g., MIT, Apache 2.0, etc.)*

---



