# MCP (Model Context Protocol) Integration Example

This example demonstrates how to use external MCP servers as tool providers with RLLM's `AgentExecutionEngine`. It runs HotpotQA question-answering evaluation using the Tavily MCP server for web search capabilities.

## Dependencies

```bash
# Install MCP CLI (if needed for other MCP servers)
uv pip install mcp


## Files

- `run_tool_mcp.py` - Main script running HotpotQA evaluation with Tavily MCP server
- `prepare_hotpotqa_data.py` - Dataset preparation script for HotpotQA

## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B \
    --dp-size 1 \
    --dtype bfloat16
```

The server should be accessible at `http://localhost:30000/v1`

## Setup

### Prerequisites

1. **Python environment** with RLLM installed
2. **Model server** running on port 30000 (see Model Hosting above)
3. **Node.js** (for Tavily MCP server)
4. **Tavily API key** from [https://app.tavily.com/home](https://app.tavily.com/home)

## Usage

```bash
# Get your Tavily API key and run the evaluation
python run_tool_mcp.py your-tavily-api-key-here
```

This will:
1. Start the Tavily MCP server via npx
2. Load the first 10 HotpotQA questions 
3. Run parallel evaluation using web search tools
4. Save results to `./trajectories/mcp_tavily/`

## Architecture

### Key Components

- **`MCPToolAgent`** - ToolAgent that works with MCP tools
- **`MCPEnvironment`** - Environment that manages MCP server connections and tool execution
- **`MCPConnectionManager`** - Handles MCP server lifecycle and tool discovery

### Integration with RLLM

The example follows standard RLLM patterns:
- Uses `AgentExecutionEngine` for parallel execution
- Uses `ToolAgent` for agent logic with search system prompt
- Integrates with RLLM's evaluation pipeline and reward system
- Saves trajectories in standard format

## Dataset

Uses HotpotQA dataset for multi-hop question answering that requires web search. The dataset is automatically downloaded and processed on first run.
