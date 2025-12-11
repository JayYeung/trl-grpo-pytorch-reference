# GRPO CPU Training - Quick Reference

## Setup (One Time)

```bash
# Make setup script executable
chmod +x setup_env.sh

# Run setup (creates venv and installs dependencies)
./setup_env.sh

# Activate environment
source grpo_cpu_env/bin/activate

# Test environment
python test_environment.py
```

## Running Training

```bash
# Make sure environment is activated
source grpo_cpu_env/bin/activate

# Run training
python cpu_grpo_qwen3_0_6b.py
```

## Quick Tweaks for Faster Testing

Edit `cpu_grpo_qwen3_0_6b.py` and modify the `Config` class:

```python
@dataclass
class Config:
    # Test with fewer samples
    max_samples: int = 10  # Default: 100

    # Fewer generations per prompt
    num_generations: int = 2  # Default: 4

    # Shorter completions
    max_completion_length: int = 64  # Default: 128
```

## Common Commands

```bash
# Activate environment
source grpo_cpu_env/bin/activate

# Deactivate environment
deactivate

# Check installed packages
pip list

# Reinstall dependencies
pip install -r requirements.txt

# Clean output directory
rm -rf qwen-grpo-cpu-output/
```

## Monitoring Training

Training outputs progress logs showing:

-   Current step
-   Loss values
-   Rewards
-   KL divergence

Output saved to: `qwen-grpo-cpu-output/`

## Troubleshooting

**Problem**: Out of memory
**Solution**: Reduce `max_samples`, `num_generations`, or `max_completion_length`

**Problem**: Import errors
**Solution**: Make sure environment is activated: `source grpo_cpu_env/bin/activate`

**Problem**: Too slow
**Solution**: This is expected on CPU. Reduce dataset size or use GPU/Neuron for real training.
