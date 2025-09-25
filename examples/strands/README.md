# rLLM × Strands Integration

This directory contains the integration between rLLM and Strands Agents SDK, enabling reinforcement learning workflows with powerful agent capabilities.

## Overview

The integration provides a **hybrid architecture** that:
- ✅ **Unifies sampling/training policies**: Strands agents use the same RolloutEngine for both inference and training
- ✅ **Maintains standard Strands architecture**: Tool execution handled by Strands event loop
- ✅ **Enables complete RL trajectory tracking**: Full tool call information captured for training
- ✅ **Supports multiple engines**: OpenAIEngine (current) with VerlEngine support planned

## Architecture

### Hybrid Design

The integration uses a **hybrid architecture** that maintains Strands SDK compliance while enabling complete RL trajectory tracking:

### Hybrid Solution Design

**Standard Strands Architecture**:
```
Agent (handles event loop and tool execution)
  ↓ calls
Model (only handles model interaction and streaming events)
  ↓ calls  
Underlying Model API (OpenAI, Anthropic, etc.)
```

```
StrandsAgent
├── Inherits standard Strands Agent
├── Uses RLLMModel for inference
├── Records complete trajectory data (for RL)
├── Receives tool call info from Model
└── Lets Strands event loop handle tool execution

RLLMModel  
├── Inherits abstract Model
├── Interacts with RolloutEngine only
├── Generates standard StreamEvents for Strands
├── Passes tool call info to Agent (trajectory tracking)
└── Does NOT execute tools (maintains separation)
```

### Design Principles

1. **Model Layer**: Only handles RolloutEngine interaction and StreamEvent generation
2. **Agent Layer**: Handles tool execution, trajectory tracking, and event loop
3. **Clear Separation**: Model doesn't execute tools, Agent doesn't call RolloutEngine directly
4. **Information Passing**: Model notifies Agent about tool calls for RL trajectory tracking

### Key Benefits

- **Policy Alignment**: Same model used for sampling and training (eliminates off-policy risk)
- **Architecture Compliance**: Follows Strands SDK standards
- **Complete Trajectories**: Full RL-ready data including tool calls
- **Maintainability**: Clear responsibilities and separation of concerns
- **Scalability**: Can be extended to other agent frameworks

## Quick Start

### Basic Usage

```python
from rllm.engine.rollout import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools import calculator

# Create RolloutEngine
engine = OpenAIEngine(
    model="gpt-4o",
    api_key="your-key",
    sampling_params={"temperature": 0.7}
)

# Create Model and Agent
model = RLLMModel(rollout_engine=engine)
agent = StrandsAgent(
    model=model,
    tools=[calculator.calculator],
    system_prompt="You are a helpful assistant with access to tools."
)

# Use the agent
result = agent("What is 15 * 23 + 67?")
trajectory = agent.trajectory  # Complete RL trajectory data
```

### Running Examples

```bash
# Simple example with inferencing/training episodes with tool_call too
python run_strands.py

# Complex research task with multiple tools
STRANDS_TASK="Research task..." python run_strands.py

# Evaluation on GAIA benchmark
cd eval/gaia && python run_gaia_eval.py
python run_gaia_eval.py --max_samples 2 # if you don't want to run the full dataset


## Implementation Details

### RLLMModel Features
- **Standard Model Interface**: Implements Strands `Model` abstract class
- **Tool Spec Conversion**: Converts various tool formats to OpenAI-compatible specs
- **Stream Event Generation**: Produces proper `toolUse` events for Strands event loop
- **Trajectory Integration**: Passes tool call info to Agent without executing tools

### StrandsAgent Features  
- **Trajectory Tracking**: Records complete `Step` objects with tool call details
- **Standard Agent Interface**: Full compatibility with Strands Agent API
- **Tool Call Recording**: Hybrid approach captures tool info for RL training
- **Chat Completion Export**: Converts Strands messages to training format

### Tool Integration
Supports multiple tool formats:
- `@tool` decorator (modern Strands tools)
- `TOOL_SPEC` format (native Strands)
- Function-based tools
- Custom wrapped tools

## File Structure

```
examples/strands/
├── README.md                 # This file
├── run_strands.py           # Main example script  
├── strands_workflow.py      # Workflow integration
├── gsearch_tool_wrapped.py  # Example custom tool
├── eval/gaia/              # GAIA benchmark evaluation
└── outputs/                # Generated trajectories and results
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY` / `TOGETHER_AI_API_KEY`: API credentials
- `STRANDS_TASK`: Task description for the agent
- `SYSTEM_PROMPT`: Custom system prompt
- `MODEL_NAME`: Model identifier

### Trajectory Output
All agent interactions are saved as JSON trajectories in `strands_outputs/`:

```json
{
  "task": "What is 303 * 12314?",
  "steps": [
    {
      "step_type": "tool_calls",
      "tool_calls": [
        {
          "name": "calculator",
          "input": {"expression": "303 * 12314"},
          "id": "call_123"
        }
      ],
      "final_response": "The result is 3,731,142"
    }
  ],
  "metadata": {
    "total_tool_calls": 1,
    "agent_type": "StrandsAgent"
  }
}
```

## Development

### Testing
```bash
# Run basic functionality test
python run_strands.py

# Run with complex multi-tool task  
STRANDS_TASK="Complex research task" python run_strands.py

# Verify trajectory output
ls strands_outputs/
```

## Future Improvements

### Architecture Evolution
- **Multi-SDK Support**: Extend to LangChain, AutoGen
- **Advanced RL Features**: Reward modeling, policy optimization
- **Enterprise Features**: Logging, monitoring, scaling

## Contributing

1. Follow the hybrid architecture pattern
2. Maintain Strands SDK compatibility  
3. Ensure complete trajectory tracking
4. Add comprehensive tests
5. Update documentation

## Related Files

- `rllm/integrations/strands.py` - Core integration implementation
- `rllm/engine/rollout.py` - RolloutEngine interface
- `rllm/agents/agent.py` - Trajectory and Step classes

---

*This integration successfully implements a hybrid architecture that maintains Strands SDK compliance while enabling complete RL trajectory tracking.*