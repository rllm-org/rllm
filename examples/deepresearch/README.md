# rLLM × Tongyi DeepResearch Integration

This integration ports Tongyi DeepResearch's multi-turn ReAct agent to work with rLLM's AgentWorkflowEngine, enabling parallel execution and trajectory tracking while preserving the original research capabilities.

## Key Implementation

### Multi-turn ReAct Agent (`deepresearch_agent.py`)
- **Ported from original**: 95% code reuse from DeepResearch's `react_agent.py`
- **rLLM Integration**: Uses `OpenAIEngine` instead of original server calls
- **Multi-turn Loop**: Maintains thinking → tool calling → observation → reasoning cycle
- **Tool Calling**: JSON-based tool calls with `<tool_call>` format, compatible with rLLM

### Workflow Wrapper (`deepresearch_workflow.py`)
- **AgentWorkflowEngine Compatible**: Inherits from `Workflow` base class
- **Episode Conversion**: Converts DeepResearch conversation history to rLLM `Episode`/`Trajectory` format
- **Parallel Execution**: Enables high-performance parallel research tasks via AgentWorkflowEngine
- **Stateless**: Each workflow instance manages independent task execution

### Real Research Tools (`deepresearch_tools.py`)
- **Serper API Search**: Real web search using same API as original DeepResearch
- **Tool Interface**: Compatible with both DeepResearch JSON format and rLLM tool calling
- **Async Support**: All tools implement async `call()` method for rLLM compatibility

## Quick Start

### Setup
```bash
conda activate rllm
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# SERPER_KEY_ID=your_serper_key  # Get free key from serper.dev
```

### Run Evaluation
```bash
# Single task test
python run_deepresearch_eval.py --dataset sample --max-samples 1

# GAIA dataset evaluation
python run_deepresearch_eval.py --dataset gaia --gaia-path path/to/gaia.json --max-samples 10
```

### Custom Model Endpoints
```bash
# Together AI
python run_deepresearch_eval.py --model Qwen/Qwen2.5-7B-Instruct-Turbo --base-url https://api.together.xyz/v1

# vLLM hosting
python run_deepresearch_eval.py --model your-model --base-url http://your-server:8000/v1
```

## Architecture Flow

```
User Question → AgentWorkflowEngine → DeepResearchWorkflow → MultiTurnReactAgent
                      ↓                        ↓                      ↓
              Parallel Execution    Episode Conversion      ReAct Loop (thinking→tool→observation)
                      ↓                        ↓                      ↓
              Episode/Trajectory ←── rLLM Format ←────── Tool Calls (Search, Python, etc.)
```

## Key Benefits

- ✅ **Original Logic Preserved**: Complete ReAct reasoning patterns from DeepResearch
- ✅ **rLLM Integration**: Full compatibility with AgentWorkflowEngine for parallel execution
- ✅ **Real Research Capabilities**: Serper API web search, Python execution, file parsing
- ✅ **Flexible Model Support**: Works with OpenAI, Together AI, or custom vLLM endpoints
- ✅ **Trajectory Tracking**: Complete conversation history for RL training

## Files

- `deepresearch_agent.py` - Multi-turn ReAct agent (ported from original)
- `deepresearch_workflow.py` - rLLM workflow wrapper
- `deepresearch_tools.py` - Research tools with real API integrations
- `run_deepresearch_eval.py` - Evaluation script with AgentWorkflowEngine
- `react_agent_original.py` - Original reference implementation
- `tool_*_original.py` - Original tool references

## Configuration

**API Keys (required):**
- `OPENAI_API_KEY` - OpenAI/compatible model API
- `SERPER_KEY_ID` - Web search API (free at serper.dev)

**Model Options:**
- `OPENAI_BASE_URL` - Custom endpoint for vLLM hosting
- `MODEL_NAME` - Model identifier
- `TOGETHER_AI_API_KEY` - Alternative to OpenAI

## Implementation Notes

**Multi-turn Compatibility:**
- Each `workflow.run()` call creates a fresh agent instance
- Conversation state maintained in agent's message list
- Tool calls executed asynchronously with proper error handling
- Episode created from final conversation history

**Tool Integration:**
- Tools implement both DeepResearch JSON format and rLLM async interface
- Search tool uses identical Serper API logic as original
- Tool responses formatted consistently for model consumption

**AgentWorkflowEngine Integration:**
- Workflow inherits from `Workflow` base class
- No registered agents needed - workflow manages its own agent
- Episode construction converts DeepResearch results to rLLM format
- Parallel execution via workflow pool management

---

*This integration successfully ports DeepResearch's 30.5B parameter research capabilities to rLLM's infrastructure while maintaining full compatibility with the original reasoning patterns.*