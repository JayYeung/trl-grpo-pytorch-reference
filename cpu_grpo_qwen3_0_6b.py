#!/usr/bin/env python3
"""
CPU-only GRPO reference implementation for Qwen3-0.6B using TRL.

This script provides a reference implementation that mirrors the GRPO flow
from optimum-neuron-grpo but runs entirely on CPU with vanilla PyTorch.

Based on: https://github.com/alex1xu/optimum-neuron-grpo/tree/alex-save-wip4
"""

import os
import re
import csv
from dataclasses import dataclass
from typing import List
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration for CPU GRPO training - keeps values small for local testing"""
    model_name: str = "Qwen/Qwen2.5-0.5B"  # Using Qwen2.5-0.5B as closest to 0.6B
    
    # Dataset settings
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    max_samples: int = 10  # Limit samples for CPU testing (10 samples = ~100 steps with 5 epochs)
    
    # Token length limits (keep small for CPU)
    max_prompt_length: int = 128
    max_completion_length: int = 64
    
    # Training hyperparameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    num_train_epochs: int = 2
    
    # GRPO-specific parameters
    num_generations: int = 2        # Group size G (number of completions per prompt) - reduced for speed
    beta: float = 0.01              # KL divergence coefficient (called 'beta' in TRL)
    
    # Output and logging
    output_dir: str = "qwen-grpo-cpu-output"
    logging_steps: int = 5
    save_steps: int = 10000
    seed: int = 42


cfg = Config()


# -----------------------------------------------------------------------------
# 2. Dataset preparation - GSM8K math problems
# -----------------------------------------------------------------------------

def prepare_gsm8k_dataset(max_samples: int = None) -> Dataset:
    """
    Load GSM8K dataset and format for GRPO training.
    
    GSM8K contains grade school math word problems with solutions.
    We'll use the questions as prompts.
    """
    print(f"Loading GSM8K dataset from {cfg.dataset_name}...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Limited to {len(ds)} samples for CPU testing")
    
    def format_prompt(example):
        """Format the question as a prompt for the model"""
        question = example["question"]
        # Add instruction prefix to guide the model
        prompt = f"Solve this math problem step by step:\n\nQuestion: {question}\n\nAnswer:"
        return {"prompt": prompt, "ground_truth": example["answer"]}
    
    formatted_ds = ds.map(format_prompt, remove_columns=ds.column_names)
    print(f"Prepared {len(formatted_ds)} training examples")
    
    return formatted_ds


# -----------------------------------------------------------------------------
# 3. Reward function - Extract and verify answers
# -----------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """
    Extract numeric answer from text - searches anywhere in the text.
    
    Matches the Neuron implementation's logic:
    1. First try to find answer after #### marker (GSM8K format)
    2. If no #### marker, just find any number in the text
    """
    # First try to find answer after #### marker (GSM8K format)
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        num_match = re.search(r"[\d,]+(?:\.\d+)?", match.group(1).strip())
        if num_match:
            return num_match.group(0).replace(",", "")
    
    # If no #### marker, just find any number in the text
    num_match = re.search(r"[\d,]+(?:\.\d+)?", text)
    if num_match:
        return num_match.group(0).replace(",", "")
    
    return ""


def extract_all_numbers(text: str) -> list[str]:
    """Extract all numbers from text."""
    return [match.group(0).replace(",", "") for match in re.finditer(r"[\d,]+(?:\.\d+)?", text)]


def create_reward_function(dataset):
    """
    Create reward function that checks if generated answer matches ground truth.
    
    This matches the Neuron implementation's reward logic:
    - Compares extracted numbers from completions against ground truth answers
    - Returns 1.0 for correct answers, 0.0 otherwise
    
    Args:
        dataset: Dataset with 'prompt' and 'ground_truth' columns
        
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    # Build cache of prompt -> ground truth answer
    answer_cache = {}
    for item in dataset:
        prompt = item["prompt"]
        ground_truth = item.get("ground_truth", "")
        answer_cache[prompt] = ground_truth
    
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Reward function for GRPO training.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            **kwargs: Additional arguments passed by TRL
            
        Returns:
            List of reward scores (1.0 for correct, 0.0 for incorrect)
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            # Get ground truth for this prompt
            gt = answer_cache.get(prompt)
            
            if not gt:
                # No ground truth available
                rewards.append(0.0)
                continue
            
            # Extract correct answer from ground truth
            gt_ans = extract_answer(gt)
            
            if not gt_ans:
                # Could not extract answer from ground truth
                rewards.append(0.0)
                continue
            
            # Extract all numbers from completion
            all_numbers = extract_all_numbers(completion)
            
            # Check if the correct answer appears anywhere in the completion
            if gt_ans in all_numbers:
                rewards.append(1.0)  # Correct!
            else:
                rewards.append(0.0)  # Incorrect
        
        return rewards
    
    return reward_fn


def create_exact_match_reward_function(dataset):
    """
    Create reward function with similarity-based matching.
    
    This is the enhanced version from the Neuron implementation that:
    - Rewards exact matches with 1.0
    - Rewards partial matches based on containment and similarity
    - Uses difflib for fuzzy matching
    
    Args:
        dataset: Dataset with 'prompt' and 'ground_truth' columns
        
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    # Build cache of prompt -> ground truth answer
    answer_cache = {item["prompt"]: item.get("ground_truth", "").strip() for item in dataset}

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Similarity-based reward function for GRPO training.
        
        Reward calculation:
        - 1.0: Exact match
        - 0.5-1.0: Ground truth contained in completion (scaled by length ratio)
        - 0.3-0.8: Completion contained in ground truth (scaled by length ratio)
        - 0.0-0.7: Fuzzy similarity match (similarity - 0.2, capped at 0.9 * similarity)
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            **kwargs: Additional arguments (e.g., completion_ids) passed by TRL
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        from difflib import SequenceMatcher
        
        rewards = []
        ground_truths = []

        for prompt, completion in zip(prompts, completions):
            gt = answer_cache.get(prompt)
            if gt is None:
                rewards.append(0.0)
                ground_truths.append("")
                continue

            gt = gt.strip()
            ground_truths.append(gt)
            comp = completion.strip()

            # Calculate string similarity
            similarity = SequenceMatcher(None, comp, gt).ratio()
            
            # Calculate reward based on matching type
            if comp == gt:
                # Perfect match
                reward = 1.0
            elif gt in comp:
                # Ground truth is contained in completion
                ratio = len(gt) / len(comp)
                reward = 0.5 + 0.5 * ratio
            elif comp in gt:
                # Completion is contained in ground truth
                ratio = len(comp) / len(gt)
                reward = 0.3 + 0.5 * ratio
            else:
                # No containment, use similarity with penalty
                reward = max(0.0, similarity - 0.2)
            
            # Take the maximum of calculated reward and similarity-based reward
            reward = max(reward, similarity * 0.9)
            rewards.append(reward)

        return rewards

    return reward_fn


# -----------------------------------------------------------------------------
# 4. Model and Tokenizer Setup
# -----------------------------------------------------------------------------

def setup_model_and_tokenizer():
    """Load Qwen model and tokenizer for CPU training"""
    print(f"\nLoading model: {cfg.model_name}")
    print("Note: This will download ~1GB on first run")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True
    )
    
    # Qwen models need explicit pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Load model with float32 (MPS supports float32)
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
        device_map={"":device_str},
        trust_remote_code=True
    )
    
    print(f"Model loaded on {device_str.upper()} with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    return model, tokenizer


# -----------------------------------------------------------------------------
# 5. GRPO Training Setup
# -----------------------------------------------------------------------------

class MetricsLogger:
    """Simple CSV logger for GRPO metrics."""
    
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"grpo_metrics_{timestamp}.csv")
        
        self.fieldnames = [
            "step", "epoch", "loss", "learning_rate",
            "mean_reward", "max_reward", "min_reward", "std_reward",
            "accuracy", "num_samples",
            "example_prompt_1", "example_completion_1", "example_reward_1",
            "example_prompt_2", "example_completion_2", "example_reward_2",
        ]
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        print(f"📊 Metrics logging to: {self.csv_path}")
    
    def log(self, metrics: dict):
        """Append metrics to CSV."""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)


def create_grpo_trainer(model, tokenizer, train_dataset):
    """
    Create GRPOTrainer with configuration.
    
    This mirrors the NeuronGRPOTrainer setup but uses TRL's standard GRPOTrainer.
    """
    
    # Create GRPO configuration (analog to NeuronGRPOConfig)
    training_args = GRPOConfig(
        # Output and logging
        output_dir=cfg.output_dir,
        run_name="qwen-grpo-cpu",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        
        # Training hyperparameters
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        
        # Precision (CPU: use float32)
        bf16=False,
        fp16=False,
        
        # GRPO-specific parameters
        num_generations=cfg.num_generations,  # G completions per prompt
        generation_batch_size=cfg.num_generations,  # Must be divisible by num_generations
        beta=cfg.beta,  # KL penalty for staying close to reference policy
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        sync_ref_model=False,  # Don't load a separate reference model (saves memory on CPU)
        
        # Reproducibility
        seed=cfg.seed,
        
        # Optimization
        gradient_checkpointing=False,  # Disabled for simplicity on CPU
        optim="adamw_torch",
        
        # Reporting
        report_to="none",  # Disable wandb/tensorboard for simple local runs
    )
    
    print("\nGRPO Configuration:")
    print(f"  Batch size: {cfg.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    print(f"  Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  Generations per prompt: {cfg.num_generations}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  KL coefficient (beta): {cfg.beta}")
    
    # Create reward function with similarity-based matching (matches Neuron implementation)
    reward_fn = create_exact_match_reward_function(train_dataset)
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(cfg.output_dir)
    
    # Create custom trainer with metrics logging
    class MetricsTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            self.metrics_logger = kwargs.pop('metrics_logger', None)
            self.last_prompts = []
            self.last_completions = []
            self.last_rewards = None  # Will be numpy array
            super().__init__(*args, **kwargs)
        
        def training_step(self, model, inputs, num_items_in_batch=None):
            # Call parent training step first
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Always log metrics after training step
            if self.metrics_logger:
                try:
                    # Store prompts/completions for logging
                    if 'prompt_ids' in inputs and hasattr(self, 'processing_class'):
                        self.last_prompts = self.processing_class.batch_decode(
                            inputs['prompt_ids'].cpu(), skip_special_tokens=True
                        )[:2]
                    if 'completion_ids' in inputs and hasattr(self, 'processing_class'):
                        self.last_completions = self.processing_class.batch_decode(
                            inputs['completion_ids'].cpu(), skip_special_tokens=True
                        )[:2]
                    
                    if 'rewards' in inputs:
                        self.last_rewards = inputs['rewards'].cpu().detach().numpy()
                    
                    self._log_metrics(loss)
                except Exception as e:
                    print(f"Warning: Failed to log metrics: {e}")
            
            return loss
        
        def _log_metrics(self, loss):
            # Always log, use None for missing data
            
            rewards = self.last_rewards
            accuracy = (rewards >= 0.9).sum() / len(rewards) if len(rewards) > 0 else 0.0
            
            # Get learning rate
            lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
            
            metrics = {
                "step": self.state.global_step,
                "epoch": round(self.state.epoch, 4) if self.state.epoch else 0.0,
                "loss": round(float(loss.detach().cpu().item()), 6),
                "learning_rate": f"{lr:.2e}",
                "mean_reward": round(float(rewards.mean()), 4),
                "max_reward": round(float(rewards.max()), 4),
                "min_reward": round(float(rewards.min()), 4),
                "std_reward": round(float(rewards.std()), 4),
                "accuracy": round(float(accuracy), 4),
                "num_samples": len(rewards),
            }
            # Add examples
            for i in range(2):
                if self.last_prompts and i < len(self.last_prompts):
                    prompt = str(self.last_prompts[i])[:150].replace('\n', ' ')
                else:
                    prompt = None
                
                if self.last_completions and i < len(self.last_completions):
                    completion = str(self.last_completions[i])[:150].replace('\n', ' ')
                else:
                    completion = None
                
                if self.last_rewards is not None and i < len(self.last_rewards):
                    reward = round(float(self.last_rewards[i]), 4)
                else:
                    reward = None
                
                metrics[f"example_prompt_{i+1}"] = prompt
                metrics[f"example_completion_{i+1}"] = completion
                metrics[f"example_reward_{i+1}"] = reward
            
            self.metrics_logger.log(metrics)
            
            # Console log
            if mean_reward is not None:
                print(f"Step {metrics['step']}: Loss={metrics['loss']:.4f}, "
                      f"Reward={mean_reward:.4f}±{std_reward:.4f}, "
                      f"Acc={accuracy:.2%}")
            else:
                print(f"Step {metrics['step']}: Loss={metrics['loss']:.4f}, "
                      f"Reward=None (no rewards yet)")
    # Create trainer with metrics logging
    trainer = MetricsTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        metrics_logger=metrics_logger,
    )
    
    return trainer


# -----------------------------------------------------------------------------
# 6. Main Training Loop
# -----------------------------------------------------------------------------

def main():
    """Main training function"""
    print("=" * 70)
    print("CPU-based GRPO Training for Qwen3-0.6B")
    print("Reference implementation based on optimum-neuron-grpo")
    print("=" * 70)
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No CUDA
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "mps":
        print("🚀 Using Apple Silicon GPU (MPS) for acceleration")
    else:
        print("⚠️  MPS not available, falling back to CPU")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")
    
    # Prepare dataset
    print("\n" + "-" * 70)
    print("Step 1: Preparing Dataset")
    print("-" * 70)
    train_dataset = prepare_gsm8k_dataset(max_samples=cfg.max_samples)
    
    # Load model and tokenizer
    print("\n" + "-" * 70)
    print("Step 2: Loading Model and Tokenizer")
    print("-" * 70)
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create trainer
    print("\n" + "-" * 70)
    print("Step 3: Setting up GRPO Trainer")
    print("-" * 70)
    trainer = create_grpo_trainer(model, tokenizer, train_dataset)
    
    # Train
    print("\n" + "-" * 70)
    print("Step 4: Starting Training")
    print("-" * 70)
    if device.type == "cpu":
        print("\n⚠️  NOTE: Training on CPU will be slow. This is for reference/debugging.")
    else:
        print("\n✅ Training on MPS (Apple Silicon GPU) - faster than CPU!")
    print("For production, use CUDA GPU or AWS Neuron instances.\n")
    
    try:
        trainer.train()
        
        # Save final model
        print("\n" + "-" * 70)
        print("Step 5: Saving Model")
        print("-" * 70)
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"✅ Model saved to {cfg.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Partial progress saved to {cfg.output_dir}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
