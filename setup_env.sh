#!/bin/bash
# Setup script for CPU-based GRPO environment

echo "Creating Python virtual environment..."
python3 -m venv grpo_cpu_env

echo "Activating virtual environment..."
source grpo_cpu_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source grpo_cpu_env/bin/activate"
echo ""
echo "To run the training script:"
echo "  python cpu_grpo_qwen3_0_6b.py"
