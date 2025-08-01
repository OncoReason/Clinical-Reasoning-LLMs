#!/usr/bin/env python
"""
GSM8K-specific Hugging Face SFT training implementation using BaseSFTTrainer.
"""
# import torch._dynamo
# torch._dynamo.disable()

# import torch.distributed as dist
# dist.init_process_group(backend='nccl')

import re
from typing import Optional
from pydantic import BaseModel, Field
from datasets import load_dataset
from transformers import AutoTokenizer

from base_sft_trainer import BaseSFTTrainer

# import torch._dynamo
# torch._dynamo.disable()

cot_prompt_template = """ """ # Note: The prompt templates will be made publicly available upon paper acceptance.

class Answer(BaseModel):
    """Pydantic model for answer validation and extraction."""
    value: str = Field(..., description="The numerical answer value")
    reasoning: str = Field(..., description="Step-by-step reasoning")


class CustomSFTTrainer(BaseSFTTrainer):
    """Trainer for Custom dataset using BaseSFTTrainer."""

    def preprocess_data(self, examples):
        """Preprocess the dataset for training."""
        inputs = [cot_prompt_template.format(patient_data = examples["patient_data"][i], survival_status = examples["survival_status"][i], survival_months = examples["survival_months"][i], reasoning = "\n".join(examples["chain_of_thought"][i]), comment = examples["comments"][i]) for i in range(len(examples["patient_data"]))]

        model_inputs = self.tokenizer(
            inputs, max_length=2048, truncation=True, padding="max_length"
        )
        return model_inputs

    def load_data(self, split: str = "train"):
        """Load and preprocess GSM8K dataset with 80:20 split and save test set."""
        # Load full dataset
        train_data = load_dataset('json', data_files='msk_chord_cot_dataset_train_new.json', split='train') 
        test_data = load_dataset('json', data_files='msk_chord_cot_dataset_test_new.json', split='train') 

        return train_data.map(self.preprocess_data, batched=True), test_data.map(self.preprocess_data, batched=True)

    def extract_reference_answer(self, text: str) -> Optional[str]:
        """Extract reference answer from GSM8K dataset format."""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def get_system_prompt(self):
        return None


def main():
    """Main entry point for CustomSFTTrainer training."""
    
    trainer = CustomSFTTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
