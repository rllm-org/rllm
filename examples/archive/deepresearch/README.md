# DeepResearch Integration for rLLM

## Overview

This module integrates Tongyi's DeepResearch ReAct agent into the rLLM framework, enabling evaluation on academic benchmarks like HLE (Humanity's Last Exam). The integration demonstrates how to port external agent architectures into rLLM's workflow system while maintaining compatibility with the training and evaluation infrastructure.

## Architecture

```
DeepResearch Agent (ReAct with XML-based tool calling)
    ↓
DeepResearchWorkflow (rLLM Workflow wrapper)
    ↓
AgentWorkflowEngine (Parallel execution)
    ↓
Episode/Trajectory (rLLM data format)
```

### Key Components

- **`deepresearch_agent.py`**: MultiTurnReactAgent implementing Tongyi's ReAct loop with tool calling
- **`deepresearch_workflow.py`**: Wrapper that converts agent outputs to rLLM Episodes for trajectory tracking
- **`deepresearch_tools.py`**: Tool implementations (Search, Scholar, Visit, FileParser, PythonInterpreter)
- **`evaluate_hle.py`**: Evaluation script for HLE (Humanity's Last Exam) benchmark

## Installation

### Prerequisites

```bash
# Activate rLLM environment
conda activate rllm

# Install required dependencies
pip install datasets  # For HLE dataset access
pip install tiktoken  # Optional: for better token counting with OpenAI models
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# For model inference (choose one)
OPENAI_API_KEY=your_openai_key
TOGETHER_AI_API_KEY=your_together_key

# Optional: For web search tool
SERPER_API_KEY=your_serper_key  # Get free key from serper.dev
```

## Usage

### Running HLE Evaluation

```bash
# Evaluate on HLE dataset with default settings
python evaluate_hle.py --hf-dataset cais/hle --max-samples 10 --parallel-tasks 4

# Use specific model
python evaluate_hle.py --model gpt-4o --max-samples 5

# Use Together AI for evaluation
python evaluate_hle.py --model Qwen/Qwen2.5-7B-Instruct-Turbo \
                       --base-url https://api.together.xyz/v1 \
                       --max-samples 20

# Custom output directory
python evaluate_hle.py --output-dir ./my_results --max-samples 20
```

### Using DeepResearch Agent Directly

```python
from rllm.engine.rollout import OpenAIEngine
from deepresearch_agent import MultiTurnReactAgent
from deepresearch_tools import get_all_tools

# Setup rollout engine
engine = OpenAIEngine(
    model="gpt-4o",
    api_key="your_key",
    base_url="https://api.openai.com/v1"
)

# Create agent with tools
agent = MultiTurnReactAgent(
    rollout_engine=engine,
    tools=get_all_tools()
)

# Run a research task
result = await agent.run(
    question="What is the reduced 12th dimensional Spin bordism of BG2?",
    answer="Z/2"  # Optional ground truth for evaluation
)

print(f"Prediction: {result['prediction']}")
print(f"Rounds: {result['rounds']}")
print(f"Time taken: {result['time_taken']}s")
```

### Integrating with rLLM Workflows

```python
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from deepresearch_workflow import DeepResearchWorkflow

# Create workflow engine for parallel execution
workflow_engine = AgentWorkflowEngine(
    workflow_cls=DeepResearchWorkflow,
    workflow_args={
        "tools": get_all_tools(),
        "max_prompt_length": 4096,
        "max_response_length": 2048
    },
    rollout_engine=engine,
    n_parallel_tasks=4  # Run 4 tasks in parallel
)

# Run evaluation on multiple tasks
tasks = [
    {"question": "Question 1", "answer": "Answer 1"},
    {"question": "Question 2", "answer": "Answer 2"}
]

episodes = await workflow_engine.execute_tasks(tasks)

# Episodes contain full trajectories for training
for episode in episodes:
    print(f"Task: {episode.task}")
    print(f"Prediction: {episode.metrics.get('prediction')}")
    print(f"Is correct: {episode.is_correct}")
```

## Tools

The agent has access to the following research tools:

| Tool                  | Description                 | Implementation Status                |
| --------------------- | --------------------------- | ------------------------------------ |
| **Search**            | Web search via Serper API   | ✅ Fully implemented (needs API key) |
| **PythonInterpreter** | Execute Python code safely  | ✅ Fully implemented with security   |
| **Scholar**           | Academic paper search       | ❌ Placeholder only                  |
| **Visit**             | Visit and analyze web pages | ❌ Placeholder only                  |
| **FileParser**        | Parse various file formats  | ⚠️ Basic text only (no PDF/DOCX)     |

### Tool Implementation Notes

- **Search**: Real web search with Serper API integration. Configure API key in `.env` file
- **PythonInterpreter**: Enhanced security, 50s timeout, supports numpy/pandas when available
- **Scholar**: Returns placeholder results. Needs integration with arXiv/Google Scholar APIs
- **Visit**: Returns placeholder content. Needs requests/BeautifulSoup implementation
- **FileParser**: Only reads text files up to 5000 chars. Original supports PDF/DOCX/media files

## Key Improvements from Original

### 1. Token Counting Fix

- **Problem**: Original used mismatched tokenizers (GPT-2 for GPT-4o) causing incorrect context limits
- **Solution**: Now uses OpenAI API's actual token statistics from response.prompt_tokens and response.completion_tokens
- **Impact**: No more false "context exceeded" errors at 13k tokens when limit is 128k

### 2. Context Management

- **Problem**: System would incorrectly truncate messages based on wrong token counts
- **Solution**: Track actual cumulative API token consumption for accurate context management
- **Impact**: Model can use full context window effectively

### 3. System Prompt Optimization

- **Problem**: Over-constrained prompt requiring specific tags caused unnatural responses
- **Solution**: Simplified prompt matching original Tongyi design, letting model reason naturally
- **Impact**: Better convergence, fewer infinite loops

### 4. Parallel Execution

- \*\*Leverages AgentWorkflowEngine for concurrent task processing
- \*\*Configurable parallelism (n_parallel_tasks parameter)
- \*\*Automatic retry on failures

## Evaluation Results

Evaluation results will be added after running benchmarks. The system is designed to evaluate on HLE and other academic benchmarks.

## Known Issues and Limitations

1. **Tool Placeholders**: Scholar and Visit tools need real implementations for research tasks
2. **Model-Specific Behavior**:
   - Some models may not consistently use `<answer>` tags
   - Tool calling format adherence varies by model
3. **Long Context Tasks**: Very complex research may still hit token limits
4. **Judge Accuracy**: LLM judge may not perfectly evaluate complex answers

## Future Improvements

- [ ] Implement real Scholar tool using arXiv/Semantic Scholar APIs
- [ ] Implement real Visit tool using requests/BeautifulSoup
- [ ] Add PDF/DOCX parsing to FileParser
- [ ] Create unified evaluation framework for multiple benchmarks
- [ ] Add more Tongyi agents (QwenCoder, etc.)
- [ ] Improve judge accuracy with better prompts

## Project Structure

```
examples/deepresearch/
├── deepresearch_agent.py      # Core ReAct agent implementation
├── deepresearch_workflow.py   # rLLM workflow wrapper
├── deepresearch_tools.py      # Tool implementations
├── evaluate_hle.py            # HLE evaluation script
├── react_agent_original.py    # Original Tongyi reference
├── tool_*_original.py         # Original tool references
├── hle_outputs/              # Evaluation results (git ignored)
└── README.md                  # This file
```

## Contributing

To add new tools or improve existing ones:

1. Implement tool in `deepresearch_tools.py` following the pattern:

   ```python
   class YourTool(DeepResearchTool):
       async def call(self, **kwargs) -> str:
           # Your implementation
           return result_string
   ```

2. Add to `DEEPRESEARCH_TOOLS` registry

3. Test with evaluation script

4. Submit PR with test results

## Related Work

This integration is part of the rLLM evaluation framework initiative. See also:

- `examples/strands/` - Strands agent integration
- `rllm/agents/` - Native rLLM agents
- `rllm/workflows/` - Workflow base classes

## Citation

If you use this integration, please cite:

```bibtex
@misc{deepresearch2024,
  title={DeepResearch: Multi-turn Research Agent},
  author={Alibaba NLP Team},
  year={2024},
  url={https://github.com/Alibaba-NLP/DeepResearch}
}
```

## License

This integration follows rLLM's license. The original DeepResearch implementation is from Alibaba's Tongyi team.
