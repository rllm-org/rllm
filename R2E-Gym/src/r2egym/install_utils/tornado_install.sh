uv venv --python=python3.9
source .venv/bin/activate

uv pip install -e .
uv pip install pycares pycurl twisted tox pytest
