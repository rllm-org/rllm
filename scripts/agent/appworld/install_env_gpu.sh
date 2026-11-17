#!/bin/bash
# AppWorld Environment Setup Script (GPU Version)
# This script sets up the rLLM environment with GPU support for AppWorld

set -e  # Exit on error

echo "=========================================="
echo "AppWorld GPU Environment Setup"
echo "=========================================="

# ========== Step 1: Create conda environment ==========
echo "Step 1: Creating conda environment..."
conda create -n rllm_py311_gpu python=3.11 -y
conda activate rllm_py311_gpu

# ========== Step 2: Install PyTorch (GPU version) ==========
echo "Step 2: Installing PyTorch with CUDA support..."
# Use PyTorch 2.7.0 + CUDA 12.1 (adjust according to your CUDA version)
# If your CUDA version is 11.8, change to pytorch-cuda=11.8
conda install pytorch=2.7.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify PyTorch and CUDA
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# ========== Step 3: Install other core packages (conda) ==========
echo "Step 3: Installing core packages..."
conda install -c conda-forge \
    numpy scipy pandas pyarrow \
    scikit-learn nltk pyyaml pydantic \
    pytest gymnasium -y

# ========== Step 4: Install API and utility packages (conda) ==========
echo "Step 4: Installing API and utility packages..."
conda install -c conda-forge \
    httpx openai tabulate fire -y

# ========== Step 5: Install ML packages (use conda when available) ==========
echo "Step 5: Installing ML packages..."
conda install -c conda-forge \
    transformers datasets polars -y

# ========== Step 6: Install rLLM ==========
echo "Step 6: Installing rLLM..."
python -m pip install -e .

# Verify rLLM installation
echo "Verifying rLLM installation..."
python -c "
import sys
import torch
import numpy as np
import transformers
import openai
from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.appworld.appworld_env import AppWorldEnv

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('✅ Python:', sys.version.split()[0])
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA Version:', torch.version.cuda)
    print('✅ GPU Count:', torch.cuda.device_count())
    print('✅ GPU Name:', torch.cuda.get_device_name(0))
print('✅ NumPy:', np.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ OpenAI:', openai.__version__)
print('✅ All rLLM components imported successfully!')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
"

# ========== Step 7: Install AppWorld ==========
echo "Step 7: Installing AppWorld..."
python -m pip install git+https://github.com/StonyBrookNLP/appworld.git

# Verify AppWorld
echo "Verifying AppWorld installation..."
python -c "import appworld; print(f'✅ AppWorld version: {appworld.__version__}'); print(f'✅ Python version: {__import__(\"sys\").version.split()[0]}')"

# ========== Step 8: Configure AppWorld data ==========
echo "Step 8: Setting up AppWorld data..."
echo "Running: appworld install"
appworld install

echo ""
echo "NOTE: Please manually run the following command to download data (specify path):"
echo "  appworld download data --root /path/to/your/data/directory"
echo ""
echo "Then verify task data:"
echo "  appworld verify tasks"

# ========== Optional: Debugging commands ==========
echo ""
echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "To activate this environment in the future, run:"
echo "  conda activate rllm_py311_gpu"
echo ""
echo "Optional: Run the following for debugging:"
echo "  # Print available AppWorld functions"
echo "  python -c \"import appworld; print([x for x in dir(appworld) if not x.startswith('_')])\""
echo ""
echo "  # Test loading a task"
echo "  python -c \"from appworld import load_task_ids, AppWorld; task_id = load_task_ids('dev')[0]; app = AppWorld(task_id=task_id); print(f'Instruction: {app.task.instruction}')\""

