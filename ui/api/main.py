"""rLLM UI API - FastAPI Application.

Main entry point for the API backend.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import health, sessions, metrics, sse, episodes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    init_db()
    yield
    # Shutdown (nothing to do)


# Create FastAPI app
app = FastAPI(
    title="rLLM UI API",
    description="Backend API for rLLM training monitoring and visualization",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(metrics.router)
app.include_router(sse.router)
app.include_router(episodes.router)

