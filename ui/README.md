# rLLM UI

Web interface for monitoring and visualizing rLLM training runs.

## Features

- **Real-time Training Monitoring**: Live metrics and progress tracking via Server-Sent Events
- **Episode Visualization**: View trajectory details with multi-agent workflows
- **Source Code Display**: See workflow and reward function implementations
- **Progress Tracking**: Visual progress bar with epoch/batch information

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install rLLM with UI dependencies
pip install -e ".[ui]"

# Install frontend dependencies
cd ui/frontend
npm install
```

### Running

**Terminal 1 - Backend API:**
```bash
cd ui/api
uvicorn main:app --reload --port 3000
```

**Terminal 2 - Frontend:**
```bash
cd ui/frontend
npm run dev
```

Open http://localhost:5173

## Database Setup

The UI supports two database backends: **SQLite** (default) and **PostgreSQL**.

### SQLite (Default)

SQLite requires no additional setup. The database file (`rllm_ui.db`) is created automatically in the `ui/api/` directory.

```bash
# Just start the server - SQLite is used by default
cd ui/api
uvicorn main:app --reload --port 3000
```

**Pros:** Zero configuration, great for local development
**Cons:** Basic substring search only (no stemming)

### PostgreSQL

PostgreSQL provides advanced full-text search with stemming and relevance ranking.

#### Option 1: Docker (Recommended)

```bash
# Start PostgreSQL container
docker run -d \
  --name rllm-postgres \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=rllm \
  -p 5433:5432 \
  postgres:15

# Start the backend with PostgreSQL
cd ui/api
DATABASE_URL="postgresql://postgres:secret@localhost:5433/rllm" uvicorn main:app --reload --port 3000
```

#### Option 2: Local PostgreSQL

```bash
# Create database
createdb rllm

# Start with local PostgreSQL
DATABASE_URL="postgresql://localhost/rllm" uvicorn main:app --reload --port 3000
```

#### Option 3: Environment File

Create a `.env` file in `ui/api/`:

```bash
# ui/api/.env
DATABASE_URL=postgresql://postgres:secret@localhost:5433/rllm
```

Then start normally:
```bash
cd ui/api
uvicorn main:app --reload --port 3000
```

### Search Feature Comparison

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Substring matching | ✓ | ✓ |
| Stemming ("subtract" matches "subtraction") | ✗ | ✓ |
| Relevance ranking | ✗ | ✓ |
| Boolean queries | ✗ | ✓ |

### Verify Database Connection

Check which database is active:
```bash
curl http://localhost:3000/api/health
# Returns: {"status": "ok", "datastore": "SQLiteStore"} or "PostgresStore"
```

### Using with Training

To enable UI logging during training, add `"ui"` to your logger backend:

```python
tracking_logger = Tracking(
    project_name="my_project",
    experiment_name="my_experiment",
    default_backend=["wandb", "ui"],  # Add "ui" here
    config=config,
)
```

Or in your training config:
```yaml
trainer:
  logger_backend:
    - wandb
    - ui
```

## Architecture

```
ui/
├── api/          # FastAPI server (metrics, episodes, sessions)
└── frontend/     # React + TypeScript UI
```

The UI integrates with rLLM via the `UILogger` class in `rllm/utils/tracking.py`, which sends training metrics and episode data to the API server for visualization.


## API Endpoints

- `GET /api/sessions` - List training sessions
- `GET /api/sessions/{id}` - Get session details
- `GET /api/metrics/{session_id}` - Get metrics for a session
- `GET /api/episodes/{session_id}` - Get episodes for a session
- `GET /api/sse/{session_id}` - Server-Sent Events for real-time updates
