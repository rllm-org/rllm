uv venv --python=python3.8
source .venv/bin/activate

uv pip install -e .
uv pip install "pyramid[testing]"
uv pip install tox pytest