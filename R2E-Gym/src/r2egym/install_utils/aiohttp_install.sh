uv venv --python 3.9
source .venv/bin/activate

make .develop

uv pip install pytest pytest-asyncio pytest-cov pytest-asyncio pytest-mock coverage gunicorn async-generator brotlipy cython multdict yarl async-timeout trustme chardet

.venv/bin/python process_aiohttp_updateasyncio.py