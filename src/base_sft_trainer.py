"""
Base trainer implementation for SFT fine-tuning.

This abstract class provides a template for supervised fine-tuning (SFT) trainers.
It handles argument parsing, PEFT and quantization setup, model/tokenizer loading,
and the training loop. Subclasses must implement data loading and system prompt logic.
"""

import torch
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig

from trl import ModelConfig, TrlParser, ScriptArguments, SFTConfig, SFTTrainer
from peft import LoraConfig 
from common import Completion

from huggingface_hub import login


class BaseSFTTrainer(ABC):
    """
    Abstract base class for supervised fine-tuning (SFT) trainers.

    Handles argument parsing, PEFT and quantization setup, model/tokenizer loading,
    and the training loop. Subclasses must implement data loading and system prompt logic.
    """

    def __init__(self):
        """
        Initialize the base trainer, parse arguments, and set up configs.
        """
        self.parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
        self.script_args, self.training_args, self.model_args = self.parser.parse_args_and_config()
        self.peft_config = None
        self.bnb_config = None
        self.quantization = "4bit" # Use 4bit quantization if needed
        self.hf_token = "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"  # Hugging Face token (replace with your own)
        self.cache_directory = "/scratch/rh3884/huggingface_models"
        self.use_peft = True
        self.use_bnb = False
        login(self.hf_token)  # Authenticate with Hugging Face Hub
        
    @abstractmethod
    def load_data(self, split: str = "train") -> Dataset:
        """
        Load and preprocess dataset.

        Args:
            split (str): Dataset split to load (default: "train").

        Returns:
            Dataset: The loaded and preprocessed dataset.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get system prompt for the specific task.

        Returns:
            str: The system prompt string.
        """
        pass

    def setup_peft(self) -> None:
        """
        Setup PEFT (Parameter-Efficient Fine-Tuning) configuration.

        Uses LoRA for efficient adaptation of large models.
        """
        if self.use_peft:
            self.peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],  # Adjust based on model architecture
                task_type="CAUSAL_LM"
            )
    
    def setup_bnb(self) -> None:
        """
        Setup BNB (Bits and Bytes) configuration for quantization.

        Supports 4-bit and 8-bit quantization for memory-efficient training.
        """
        if self.use_bnb:
            if self.quantization == "8bit":
                self.bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            elif self.quantization == "4bit":
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """
        Initialize and configure the tokenizer.

        Returns:
            AutoTokenizer: The loaded tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def setup_model(self) -> AutoModelForCausalLM:
        """
        Initialize and configure the model.

        Returns:
            AutoModelForCausalLM: The loaded model.
        """        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            quantization_config=self.bnb_config,
            # trust_remote_code=True,
            use_cache=False,
            use_auth_token=self.hf_token,
            cache_dir=self.cache_directory
        )
        return model
    
    def train(self) -> None:
        """
        Execute the training process.

        Loads data, sets up model and tokenizer, configures PEFT/quantization,
        and launches the SFTTrainer.
        """
        # Setup tokenizer and model
        self.tokenizer = self.setup_tokenizer()

        # Load dataset (should return (train_dataset, eval_dataset))
        dataset, eval_dataset = self.load_data()
        
        self.setup_peft()  # Setup PEFT if enabled
        self.setup_bnb()   # Setup BNB quantization if enabled
        model = self.setup_model()

        # Initialize the SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=self.training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
        )

        # Train and save the model
        trainer.train()
        trainer.save_model(self.training_args.output_dir)