#!/usr/bin/env python3
"""
Quick test script to verify the environment setup and imports.
Run this after setting up the environment to ensure everything is working.
"""

import sys

print("Testing GRPO CPU Environment Setup...")
print("=" * 60)

# Test Python version
print(f"\nPython version: {sys.version}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print("✅ Python version OK")

# Test imports
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")
    sys.exit(1)

try:
    import trl
    print(f"✅ TRL {trl.__version__}")
except ImportError as e:
    print(f"❌ TRL import failed: {e}")
    sys.exit(1)

try:
    import datasets
    print(f"✅ Datasets {datasets.__version__}")
except ImportError as e:
    print(f"❌ Datasets import failed: {e}")
    sys.exit(1)

try:
    import accelerate
    print(f"✅ Accelerate {accelerate.__version__}")
except ImportError as e:
    print(f"❌ Accelerate import failed: {e}")
    sys.exit(1)

try:
    import peft
    print(f"✅ PEFT {peft.__version__}")
except ImportError as e:
    print(f"❌ PEFT import failed: {e}")
    sys.exit(1)

# Test TRL GRPO components
try:
    from trl import GRPOConfig, GRPOTrainer
    print("✅ TRL GRPO components available")
except ImportError as e:
    print(f"❌ TRL GRPO import failed: {e}")
    sys.exit(1)

# Check device
device = torch.device("cpu")
print(f"\n✅ Device: {device}")

# Quick tensor test
try:
    x = torch.randn(2, 3)
    y = x * 2
    print(f"✅ PyTorch operations working")
except Exception as e:
    print(f"❌ PyTorch operation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 All checks passed! Environment is ready.")
print("\nYou can now run:")
print("  python cpu_grpo_qwen3_0_6b.py")
print("=" * 60)
