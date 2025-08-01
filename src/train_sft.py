#!/usr/bin/env python
"""
GSM8K-specific Hugging Face SFT training implementation using BaseSFTTrainer.

This script defines a custom supervised fine-tuning (SFT) trainer for clinical reasoning tasks,
using patient summary data and survival information. It loads training and test datasets,
applies preprocessing, and launches model training using a base trainer class.
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

# Placeholder for the prompt template used to format input data for the model.
prompt_template = """ """  # Note: The actual prompt templates will be made publicly available upon paper acceptance.

class Answer(BaseModel):
    """
    Pydantic model for answer validation and extraction.

    Attributes:
        value (str): The numerical answer value.
        reasoning (str): Step-by-step reasoning for the answer.
    """
    value: str = Field(..., description="The numerical answer value")
    reasoning: str = Field(..., description="Step-by-step reasoning")


class CustomSFTTrainer(BaseSFTTrainer):
    """
    Trainer for custom clinical dataset using BaseSFTTrainer.

    Implements dataset-specific preprocessing and loading logic.
    """

    def preprocess_data(self, examples):
        """
        Preprocess the dataset for training.

        Args:
            examples (dict): Batch of examples from the dataset.

        Returns:
            dict: Tokenized model inputs.
        """
        # Format each example using the prompt template and relevant fields.
        inputs = [
            prompt_template.format(
                patient_data=examples["patient_data"][i],
                survival_status=examples["survival_status"][i],
                survival_months=examples["survival_months"][i]
            )
            for i in range(len(examples["patient_data"]))
        ]

        # Tokenize the formatted inputs for the model.
        model_inputs = self.tokenizer(
            inputs, max_length=2048, truncation=True, padding="max_length"
        )
        return model_inputs

    def load_data(self, split: str = "train"):
        """
        Load and preprocess the training and test datasets.

        Args:
            split (str): Dataset split to load (default: "train").

        Returns:
            tuple: Preprocessed training and test datasets.
        """
        # Load training and test data from JSON files.
        train_data = load_dataset('json', data_files='msk_chord_cot_dataset_train.json', split='train') 
        test_data = load_dataset('json', data_files='msk_chord_cot_dataset_test.json', split='train') 

        # Apply preprocessing to both splits.
        return train_data.map(self.preprocess_data, batched=True), test_data.map(self.preprocess_data, batched=True)

    def extract_reference_answer(self, text: str) -> Optional[str]:
        """
        Extract the reference answer from the GSM8K dataset format.

        Args:
            text (str): The answer text containing the reference answer.

        Returns:
            Optional[str]: The extracted answer, or None if not found.
        """
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def get_system_prompt(self):
        """
        Return the system prompt for the model, if any.

        Returns:
            None: No system prompt is used in this implementation.
        """
        return None


def main():
    """
    Main entry point for CustomSFTTrainer training.

    Instantiates the trainer and starts the training process.
    """
    trainer = CustomSFTTrainer()
    trainer.train()


if __name__ == "__main__":
    main()