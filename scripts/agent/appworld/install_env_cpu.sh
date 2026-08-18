#!/bin/bash
# AppWorld Environment Setup Script (CPU Version)
# This script sets up the rLLM environment for AppWorld (with CPU for quick support)

conda create -n rllm_py311 python=3.11 -y
conda activate rllm_py311

# first  cpuonly（from pytorch channel）
conda install -c pytorch cpuonly -y

# install pytorch 2.7.0（from conda-forge）
conda install -c conda-forge pytorch=2.7.0 torchvision torchaudio libtorch -y

# verify PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# ========== Step 3: install main packages ==========
conda install -c conda-forge \
    numpy scipy pandas pyarrow \
    scikit-learn nltk pyyaml pydantic \
    pytest gymnasium -y

# ========== Step 4: install API pacakges ==========
conda install -c conda-forge \
    httpx openai tabulate fire -y

# ========== Step 5: install ML related packaged==========
conda install -c conda-forge \
    transformers datasets polars -y

python -m pip install -e .

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
print('✅ CUDA:', torch.cuda.is_available())
print('✅ NumPy:', np.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ OpenAI:', openai.__version__)
print('✅ rLLM all packaged installed')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
"


python -m pip install git+https://github.com/StonyBrookNLP/appworld.git

python -c "import appworld; print(f'✅ AppWorld version: {appworld.__version__}'); print(f'✅ Python version: {__import__(\"sys\").version.split()[0]}')"


appworld install
# Note: typer and click version should be compatible
appworld download data --root {path}

# verify the tasks are downloaded
appworld verify tasks

# below are for debugging
# print the available functions
python -c "import appworld; help(appworld); print('Available functions:'); print([x for x in dir(appworld) if not x.startswith('_')])"
python -c "import inspect;from appworld import load_task_ids; print(inspect.getsource(load_task_ids)); print(appworld.__file__)"
print(inspect.getsource(load_task_ids))

python -c "from appworld import load_task_ids, AppWorld; task_id = load_task_ids('dev')[0]; app = AppWorld(task_id=task_id); print('Task attributes:'); print([x for x in dir(app.task) if not x.startswith('_')][:20]); print(f'\\nInstruction: {app.task.instruction}')"

import inspect
from appworld import AppWorld
app = AppWorld(task_id="50e1ac9_2")
ev = app.evaluate()
print(dir(ev)) 
print(inspect.getdoc(ev))   
print(inspect.getmembers(ev, predicate=inspect.ismethod)) 