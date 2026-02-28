FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2



COPY requirements-agentcore.txt requirements-agentcore.txt
# Install from requirements file
RUN uv pip install -r requirements-agentcore.txt




RUN uv pip install aws-opentelemetry-distro==0.12.2


# Signal that this is running in Docker for host binding logic
ENV DOCKER_CONTAINER=1

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

# Copy entire project (respecting .dockerignore)
COPY . .

# Agent config: agentcore_worker.py reads these at startup.
# RLLM_AGENT_DIR adds /app/agent to sys.path so "import agent" finds agent.py.
ENV RLLM_AGENT_MODULE=agent \
    RLLM_AGENT_FUNC=rollout \
    RLLM_AGENT_DIR=/app/agent

# Use the full module path

CMD ["opentelemetry-instrument", "python", "-m", "agentcore_worker"]
