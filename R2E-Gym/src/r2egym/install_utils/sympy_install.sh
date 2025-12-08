uv venv --python=python3.8
source .venv/bin/activate && uv pip install numpy mpmath pytest ipython numexpr
source .venv/bin/activate && uv pip install -e .