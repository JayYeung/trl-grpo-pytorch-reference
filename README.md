# GRPO Reference Implementation (CPU & GPU)

A reference implementation of Group Relative Policy Optimization (GRPO) using PyTorch and Hugging Face TRL. This provides implementations for both CPU and GPU training for debugging and understanding the GRPO algorithm before deploying to AWS Neuron instances.

Based on: [optimum-neuron-grpo](https://github.com/alex1xu/optimum-neuron-grpo/tree/alex-save-wip4)

## Overview

This implementation mirrors the structure of the Neuron-based GRPO trainer but runs with vanilla PyTorch. It's designed for:

-   🔍 **Understanding GRPO**: Step through the algorithm locally
-   🐛 **Debugging**: Test changes without expensive cloud resources
-   📚 **Learning**: Reference implementation with detailed comments
-   🧪 **Experimentation**: Quick iteration on small datasets
-   🚀 **GPU Training**: Production-ready GPU implementation with bfloat16

## What's Included

-   `gpu_grpo_qwen3_0_6b.py` - **GPU training script** (recommended for production)
-   `cpu_grpo_qwen3_0_6b.py` - CPU-only training script (for debugging)
-   `requirements.txt` - Python dependencies
-   `setup_env.sh` - Automated environment setup
-   `README.md` - This file

## Quick Start

### 1. Setup Environment

The setup script will create a new virtual environment and install all dependencies:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This will:

-   Create a `grpo_cpu_env` virtual environment
-   Install PyTorch, Transformers, TRL, and other dependencies
-   Prepare everything for CPU-only training

### 2. Activate Environment

```bash
source grpo_cpu_env/bin/activate
```

### 3. Run Training

#### GPU Training (Recommended)

```bash
python gpu_grpo_qwen3_0_6b.py
```

**Features:**
- ✅ Full CUDA GPU support with automatic multi-GPU distribution
- ✅ bfloat16 precision for faster training
- ✅ Gradient checkpointing for memory efficiency
- ✅ Larger batch sizes and more samples
- ✅ Production-ready configuration

#### CPU Training (Debug Only)

```bash
python cpu_grpo_qwen3_0_6b.py
```

**Note:** Training on CPU will be slow! This is intended for reference/debugging, not production training.

## Configuration

### GPU Configuration (`gpu_grpo_qwen3_0_6b.py`)

```python
@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_samples: int = 100                   # GPU can handle more
    max_prompt_length: int = 256
    max_completion_length: int = 128
    per_device_train_batch_size: int = 4    # Larger batches on GPU
    gradient_accumulation_steps: int = 2     # Effective batch: 8
    num_generations: int = 4                 # Group size (G)
    beta: float = 0.01                       # KL penalty
    learning_rate: float = 5e-6
    # bf16=True, gradient_checkpointing=True
```

### CPU Configuration (`cpu_grpo_qwen3_0_6b.py`)

```python
@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_samples: int = 10                    # Fewer for CPU
    max_prompt_length: int = 128
    max_completion_length: int = 64
    per_device_train_batch_size: int = 1    # Minimal on CPU
    num_generations: int = 2                # Reduced for speed
    beta: float = 0.01
    learning_rate: float = 5e-6
    # bf16=False (CPU uses fp32)
```

Edit these values to experiment with different settings.

## How It Maps to Neuron Implementation

| Neuron (optimum-neuron)  | GPU Reference (TRL)          | CPU Reference (TRL)          |
| ------------------------ | ---------------------------- | ---------------------------- |
| `NeuronGRPOConfig`       | `GRPOConfig`                 | `GRPOConfig`                 |
| `NeuronGRPOTrainer`      | `GRPOTrainer`                | `GRPOTrainer`                |
| `NeuronModelForCausalLM` | `AutoModelForCausalLM`       | `AutoModelForCausalLM`       |
| `tensor_parallel_size`   | `device_map="auto"`          | N/A                          |
| `bf16=True`              | `bf16=True` (GPU)            | `bf16=False` (CPU uses fp32) |

The training loop, reward function, and dataset preparation are structurally identical across all implementations.

## Dataset: GSM8K

The script uses the GSM8K (Grade School Math 8K) dataset:

-   8,000+ grade school math word problems
-   Each with a natural language solution
-   Answers in the format `#### <number>`

Limited to 100 samples by default for CPU testing.

## Reward Function

The default reward function (`reward_fn`) uses heuristics:

-   ✅ Contains a numerical answer
-   ✅ Shows mathematical operations
-   ✅ Reasonable length (20-500 chars)
-   ✅ Structured output (step-by-step)

An enhanced version (`reward_fn_with_ground_truth`) is included for reference, which compares against correct answers.

## Project Structure

```
cuda-grpo/
├── cpu_grpo_qwen3_0_6b.py    # Main training script
├── requirements.txt           # Python dependencies
├── setup_env.sh              # Environment setup script
├── README.md                 # This file
├── grpo_cpu_env/             # Virtual environment (created by setup)
└── qwen-grpo-cpu-output/     # Training outputs (created during training)
    ├── checkpoint-*/
    ├── config.json
    └── pytorch_model.bin
```

## Troubleshooting

### Out of Memory

CPU training uses RAM, not VRAM. If you run out of memory:

1. Reduce `max_samples` (default: 100)
2. Reduce `num_generations` (default: 4)
3. Reduce `max_completion_length` (default: 128)
4. Close other applications

### Slow Training

This is expected on CPU! GRPO requires generating multiple completions per prompt (4 by default), which is computationally expensive.

To speed up testing:

-   Use fewer samples: `max_samples = 10`
-   Use fewer generations: `num_generations = 2`
-   Use shorter completions: `max_completion_length = 64`

### Import Errors

Make sure the virtual environment is activated:

```bash
source grpo_cpu_env/bin/activate
pip list  # Verify packages are installed
```

## Next Steps

### Moving to Production (GPU/Neuron)

Once you've validated your approach locally, migrate to optimum-neuron:

```python
# Replace CPU imports
from optimum.neuron import NeuronGRPOConfig, NeuronGRPOTrainer
from optimum.neuron.models.training import NeuronModelForCausalLM

# Add Neuron-specific config
training_args = NeuronGRPOConfig(
    # ... same parameters as GRPOConfig
    tensor_parallel_size=8,      # NEW: Neuron parallelism
    bf16=True,                   # NEW: Use bfloat16 on Neuron
)

# Use Neuron model
model = NeuronModelForCausalLM.from_pretrained(
    cfg.model_name,
    tensor_parallel_size=8,
)
```

### Customizing the Reward Function

To use ground truth answers from GSM8K:

1. Modify the dataset mapping to include `ground_truth`
2. Pass ground truth through the trainer (requires custom trainer loop)
3. Compare extracted answers against ground truth in `reward_fn`

See `reward_fn_with_ground_truth()` in the script for an example.

### Using a Different Dataset

Replace the `prepare_gsm8k_dataset()` function:

```python
def prepare_custom_dataset():
    ds = load_dataset("your/dataset")

    def format_prompt(example):
        return {"prompt": f"Your prompt: {example['text']}"}

    return ds.map(format_prompt, ...)
```

## References

-   **Original Neuron Implementation**: https://github.com/alex1xu/optimum-neuron-grpo/tree/alex-save-wip4
-   **TRL Documentation**: https://huggingface.co/docs/trl
-   **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
-   **GSM8K Dataset**: https://huggingface.co/datasets/openai/gsm8k
-   **Qwen Models**: https://huggingface.co/Qwen

## License

This is a reference implementation for educational purposes. Follow the licenses of the underlying libraries (TRL, Transformers, etc.) and the original optimum-neuron-grpo repository.

## Support

For questions about:

-   **This reference implementation**: Check the inline comments in `cpu_grpo_qwen3_0_6b.py`
-   **TRL/GRPO**: See [TRL documentation](https://huggingface.co/docs/trl)
-   **Neuron deployment**: See [optimum-neuron docs](https://huggingface.co/docs/optimum-neuron)

---

**Happy experimenting! 🚀**
