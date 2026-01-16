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
