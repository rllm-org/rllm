# rLLM UI API Backend

FastAPI backend for the rLLM UI monitoring platform.

## Setup

```bash
cd backend/api
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload --port 3000
```

## Testing

```bash
pytest -v
```
