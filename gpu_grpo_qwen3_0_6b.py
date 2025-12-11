#!/usr/bin/env python3
"""
GPU GRPO reference implementation for Qwen3-0.6B using TRL.

This script provides a reference implementation that mirrors the GRPO flow
from optimum-neuron-grpo but runs on CUDA GPUs with PyTorch.

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
    """Configuration for GPU GRPO training"""
    model_name: str = "Qwen/Qwen2.5-0.5B"  # Using Qwen2.5-0.5B as closest to 0.6B
    
    # Dataset settings
    dataset_type: str = "gsm8k"  # Options: "repeat" or "gsm8k"
    max_samples: int = 1000  # Number of training samples
    
    # Token length limits
    max_prompt_length: int = 512
    max_completion_length: int = 512
    
    # Training hyperparameters
    per_device_train_batch_size: int = 4  # Increased for GPU
    gradient_accumulation_steps: int = 2  # Effective batch size = 8
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    
    # GRPO-specific parameters
    num_generations: int = 4        # Group size G (number of completions per prompt)
    beta: float = 0.01              # KL divergence coefficient (called 'beta' in TRL)
    
    # Output and logging
    output_dir: str = "qwen-grpo-gpu-output-gsm8k"
    logging_steps: int = 1
    save_steps: int = 100
    seed: int = 42


cfg = Config()


# -----------------------------------------------------------------------------
# 2. Dataset preparation - Repeat token task
# -----------------------------------------------------------------------------

def load_repeat_dataset(
    split: str = "train",
    max_samples: int = None,
    *,
    num_samples: int = 5000,
    min_repeats: int = 2,
    max_repeats: int = 10,
    with_spaces: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Load repeat token dataset for GRPO training.
    
    This dataset tests the model's ability to follow instructions to repeat
    words a specific number of times, with or without spaces.
    """
    import random
    random.seed(seed)

    base_tokens = [
        "apple", "banana", "cat", "dog", "hello", "world",
        "foo", "bar", "qux", "neuron", "qwen", "tensor",
    ]

    prompts = []
    answers = []

    eval_prompts = []
    eval_answers = []
    for _ in range(100):
        token = random.choice(base_tokens)
        n = random.randint(min_repeats, max_repeats)
        use_spaces = with_spaces if with_spaces is not None else bool(random.getrandbits(1))
        
        if use_spaces:
            answer = " ".join([token] * n)
            prompt = f"Say the word '{token}' {n} times, separated by single spaces.\nAnswer:"
        else:
            answer = "".join([token] * n)
            prompt = f"Say the word '{token}' {n} times with no spaces.\nAnswer:"
        
        eval_prompts.append(prompt)
        eval_answers.append(answer)
    
    middle_samples = num_samples if max_samples is None else max(0, max_samples - 200)
    
    for _ in range(middle_samples):
        token = random.choice(base_tokens)
        n = random.randint(min_repeats, max_repeats)
        use_spaces = with_spaces if with_spaces is not None else bool(random.getrandbits(1))

        if use_spaces:
            answer = " ".join([token] * n)
            prompt = f"Say the word '{token}' {n} times, separated by single spaces.\nAnswer:"
        else:
            answer = "".join([token] * n)
            prompt = f"Say the word '{token}' {n} times with no spaces.\nAnswer:"

        prompts.append(prompt)
        answers.append(answer)
    
    prompts = eval_prompts + prompts + eval_prompts
    answers = eval_answers + answers + eval_answers

    dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})
    
    print(f"Generated {len(dataset)} repeat token examples")

    return dataset


def load_gsm8k_dataset(
    split: str = "train",
    max_samples: int = None,
) -> Dataset:
    """
    Load GSM8K math reasoning dataset for GRPO training.
    
    This dataset tests the model's ability to solve grade school math problems.
    """
    print(f"Loading GSM8K dataset (split: {split})...")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    # Format the dataset for GRPO with instructions
    def format_prompt(example):
        # Add instruction to encourage proper formatting
        prompt = f"{example['question']}\nLet's solve this step by step, then provide the final answer."
        return {
            "prompt": prompt,
            "ground_truth": example["answer"]
        }
    
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    print(f"Loaded {len(dataset)} GSM8K examples")
    return dataset


# -----------------------------------------------------------------------------
# 3. Reward function - Extract and verify answers
# -----------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """
    Extract numeric answer from text with multiple strategies.
    
    Tries in order:
    1. Answer after #### marker (GSM8K ground truth format)
    2. "The answer is: X" pattern (common model output)
    3. "#### X" pattern
    4. Last number in the text (fallback)
    """
    # Strategy 1: GSM8K format with #### marker
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        num_match = re.search(r"[\d,]+(?:\.\d+)?", match.group(1).strip())
        if num_match:
            return num_match.group(0).replace(",", "")
    
    # Strategy 2: "The answer is: X" or "answer is X" pattern
    match = re.search(r"(?:the\s+)?answer\s+is:?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Strategy 3: Last number in text (common for model outputs)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""


def extract_all_numbers(text: str) -> list[str]:
    """Extract all numbers from text."""
    return [match.group(0).replace(",", "") for match in re.finditer(r"[\d,]+(?:\.\d+)?", text)]


def create_reward_function(dataset):
    """
    Create improved reward function for GSM8K that encourages proper formatting.
    
    Reward structure:
    - 1.0: Correct answer with proper formatting ("The answer is: X" or "#### X")
    - 0.7: Correct answer as last number in completion
    - 0.3: Correct answer appears anywhere in completion
    - 0.1: Completion is non-empty and attempts reasoning
    - 0.0: Wrong answer or empty completion
    
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
        Reward function for GRPO training with partial credit.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            **kwargs: Additional arguments passed by TRL
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            # Get ground truth for this prompt
            gt = answer_cache.get(prompt)
            
            if not gt or not completion.strip():
                rewards.append(0.0)
                continue
            
            # Extract correct answer from ground truth
            gt_ans = extract_answer(gt)
            
            if not gt_ans:
                rewards.append(0.0)
                continue
            
            # Normalize answers for comparison (remove decimals if .0)
            try:
                gt_num = float(gt_ans)
                gt_ans_normalized = str(int(gt_num)) if gt_num == int(gt_num) else gt_ans
            except ValueError:
                gt_ans_normalized = gt_ans
            
            # Extract answer from completion
            completion_ans = extract_answer(completion)
            
            # Normalize completion answer
            try:
                comp_num = float(completion_ans) if completion_ans else None
                comp_ans_normalized = str(int(comp_num)) if comp_num and comp_num == int(comp_num) else completion_ans
            except ValueError:
                comp_ans_normalized = completion_ans
            
            # Check if answer is correct
            is_correct = (gt_ans_normalized == comp_ans_normalized or gt_ans == completion_ans)
            
            if is_correct:
                # Check for proper formatting - highest reward
                if re.search(r"(?:the\s+)?answer\s+is:?\s*" + re.escape(completion_ans), completion, re.IGNORECASE):
                    rewards.append(1.0)  # Perfect: correct with "The answer is: X" format
                elif "####" in completion:
                    rewards.append(1.0)  # Perfect: correct with #### format
                # Check if it's the last number (good practice)
                elif completion_ans and completion.rstrip().endswith(completion_ans):
                    rewards.append(0.8)  # Good: answer at the end
                # Check if answer is in the last 50 characters
                elif completion_ans in completion[-50:]:
                    rewards.append(0.7)  # Decent: answer near the end
                else:
                    rewards.append(0.5)  # Partial: correct but poorly formatted
            else:
                # Wrong answer - check if there's any reasoning attempt
                all_numbers = extract_all_numbers(completion)
                if gt_ans in all_numbers or gt_ans_normalized in [str(n).replace(",", "") for n in all_numbers]:
                    rewards.append(0.3)  # Partial: has correct number but extracted wrong one
                elif len(completion.strip()) > 20:  # At least attempting to reason
                    rewards.append(0.1)  # Small reward for trying
                else:
                    rewards.append(0.0)  # No useful output
        
        return rewards
    
    return reward_fn


def create_exact_match_reward_function(dataset):
    """
    Create reward function with similarity-based matching.
    
    This reward function:
    - Rewards exact matches with 1.0
    - Rewards partial matches based on containment and similarity
    - Uses difflib for fuzzy matching
    
    Args:
        dataset: Dataset with 'prompt' and 'answer' columns
        
    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    # Build cache of prompt -> answer
    _answer_cache = {item["prompt"]: item["answer"].strip() for item in dataset}

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
            **kwargs: Additional arguments passed by TRL
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        from difflib import SequenceMatcher
        
        rewards = []
        ground_truths = []

        for prompt, completion in zip(prompts, completions):
            gt = _answer_cache.get(prompt)
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
    """Load Qwen model and tokenizer for GPU training"""
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
    
    # Load model with bfloat16 for GPU efficiency
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
        trust_remote_code=True
    )
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
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
        run_name="qwen-grpo-gpu",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        
        # Training hyperparameters
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        
        # Precision (GPU: use bfloat16)
        bf16=True,
        fp16=False,
        
        # GRPO-specific parameters
        num_generations=cfg.num_generations,  # G completions per prompt
        generation_batch_size=cfg.num_generations,  # Must be divisible by num_generations
        beta=cfg.beta,  # KL penalty for staying close to reference policy
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        sync_ref_model=False,  # Don't load a separate reference model (saves memory)
        
        # Reproducibility
        seed=cfg.seed,
        
        # Optimization
        gradient_checkpointing=True,  # Enable for memory efficiency on GPU
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
    
    # Select appropriate reward function based on dataset
    if cfg.dataset_type == "gsm8k":
        print("\nUsing numeric extraction reward function for GSM8K")
        base_reward_fn = create_reward_function(train_dataset)
    else:  # repeat task
        print("\nUsing similarity-based reward function for repeat task")
        base_reward_fn = create_exact_match_reward_function(train_dataset)
    
    # Wrap reward function to capture prompts/completions
    captured_data = {'prompts': [], 'completions': [], 'rewards': []}
    
    def wrapped_reward_fn(prompts, completions, **kwargs):
        # Capture the data
        captured_data['prompts'] = prompts[:2] if len(prompts) >= 2 else prompts
        captured_data['completions'] = completions[:2] if len(completions) >= 2 else completions
        
        # Call original reward function
        rewards = base_reward_fn(prompts, completions, **kwargs)
        
        # Capture rewards
        captured_data['rewards'] = rewards[:2] if len(rewards) >= 2 else rewards
        
        return rewards
    
    reward_fn = wrapped_reward_fn
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(cfg.output_dir)
    
    # Create custom callback to capture metrics from TRL
    from transformers import TrainerCallback
    
    class MetricsCallback(TrainerCallback):
        def __init__(self, metrics_logger):
            self.metrics_logger = metrics_logger
            self.last_log = {}
            self.captured_data = None  # Will be set externally
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called when TRL logs metrics"""
            if logs is None:
                return
            
            # Store the latest log data
            self.last_log = logs
            
            # Extract rewards from TRL's logging
            reward_mean = logs.get('rewards/reward_fn/mean', logs.get('reward', None))
            reward_std = logs.get('rewards/reward_fn/std', logs.get('reward_std', None))
            
            # Get data from captured_data dict
            prompts = self.captured_data['prompts'] if self.captured_data else []
            completions = self.captured_data['completions'] if self.captured_data else []
            rewards = self.captured_data['rewards'] if self.captured_data else []
            
            if reward_mean is not None:
                # Get other metrics
                metrics = {
                    "step": state.global_step,
                    "epoch": round(state.epoch, 4) if state.epoch else 0.0,
                    "loss": round(logs.get('loss', 0.0), 6),
                    "learning_rate": f"{logs.get('learning_rate', 0.0):.2e}",
                    "mean_reward": round(reward_mean, 4),
                    "max_reward": round(max(rewards), 4) if rewards else None,
                    "min_reward": round(min(rewards), 4) if rewards else None,
                    "std_reward": round(reward_std, 4) if reward_std else None,
                    "accuracy": round(sum(1 for r in rewards if r >= 0.9) / len(rewards), 4) if rewards else None,
                    "num_samples": len(rewards) if rewards else None,
                    "example_prompt_1": str(prompts[0])[:].replace('\n', ' ') if len(prompts) > 0 else None,
                    "example_completion_1": str(completions[0])[:].replace('\n', ' ') if len(completions) > 0 else None,
                    "example_reward_1": round(float(rewards[0]), 4) if len(rewards) > 0 else None,
                    "example_prompt_2": str(prompts[1])[:].replace('\n', ' ') if len(prompts) > 1 else None,
                    "example_completion_2": str(completions[1])[:].replace('\n', ' ') if len(completions) > 1 else None,
                    "example_reward_2": round(float(rewards[1]), 4) if len(rewards) > 1 else None,
                }
                
                self.metrics_logger.log(metrics)
                if reward_std:
                    print(f"📊 Step {state.global_step}: Loss={metrics['loss']:.4f}, Reward={reward_mean:.4f}±{reward_std:.4f}")
                else:
                    print(f"📊 Step {state.global_step}: Loss={metrics['loss']:.4f}, Reward={reward_mean:.4f}")
    
    # Create callback with access to captured data
    callback = MetricsCallback(metrics_logger)
    callback.captured_data = captured_data
    
    # Create trainer with metrics logging
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Add callback to trainer
    trainer.add_callback(callback)
    
    return trainer


# -----------------------------------------------------------------------------
# 6. Main Training Loop
# -----------------------------------------------------------------------------

def main():
    """Main training function"""
    print("=" * 70)
    print("GPU-based GRPO Training for Qwen3-0.6B")
    print("Reference implementation based on optimum-neuron-grpo")
    print("=" * 70)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    print(f"\nDevice: {device}")
    print(f"🚀 Using {gpu_count} NVIDIA GPU(s) for training")
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")
    
    # Prepare dataset
    print("\n" + "-" * 70)
    print("Step 1: Preparing Dataset")
    print("-" * 70)
    
    if cfg.dataset_type == "gsm8k":
        train_dataset = load_gsm8k_dataset(split="train", max_samples=cfg.max_samples)
    elif cfg.dataset_type == "repeat":
        train_dataset = load_repeat_dataset(max_samples=cfg.max_samples)
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}. Must be 'gsm8k' or 'repeat'")
    
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
    print("\n✅ Training on CUDA GPU(s) - optimized for production use\n")
    
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
