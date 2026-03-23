<div align="center">

# rLLM

**Train your AI agents with RL. Any framework. Minimal code changes.**

[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://docs.rllm-project.com/)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/rllmproject/shared_invite/zt-3pyblo6ef-m9kqAoInI8xSyUBkpuOyXA)
[![Website](https://img.shields.io/badge/Site-%233f72af.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://rllm-project.com)
[![Blogs](https://img.shields.io/badge/Blogs-007AFF?style=for-the-badge)](https://rllm-project.com/blog)
[![X](https://img.shields.io/badge/-black?logo=X&style=for-the-badge)](https://x.com/rllm_project)

<!-- [![PyPI](https://img.shields.io/pypi/v/rllm?style=for-the-badge)](https://pypi.org/project/rllm/) -->

</div>

rLLM is an open-source framework for training AI agents with reinforcement learning. Swap in a tracked client, define a reward function, and let RL handle the rest — no matter what agent framework you use.

## Core Features

- **Works with any agent framework** — LangGraph, SmolAgent, Strands, OpenAI Agents SDK, Google ADK, or plain `openai.OpenAI`. Just swap the client. 🔌
- **Near-zero code changes** — Add `@rllm.rollout` to wrap your agent code, and rLLM traces every LLM call automatically. 🪄
- **CLI-first workflow** — Eval and train from the command line with 50+ built-in benchmarks. `rllm eval gsm8k` just works. ⚡
- **Battle-tested results** — rLLM-trained agents beat models 50x their size (4B → outperforms 235B on finance, 1.5B → surpasses O1-Preview on math). 📈
- **Multiple RL algorithms** — GRPO, REINFORCE, RLOO, rejection sampling, and more. 🧠
- **Two training backends** — `verl` for distributed multi-GPU training, `tinker` for single-machine / CPU setups. Same API either way. 🔧

Read more on our [documentation site](https://docs.rllm-project.com/).

## Installation

rLLM requires `Python >= 3.10` (`3.11` is needed if using `tinker`). You can install it either directly via pip or build from source.

```bash
uv pip install "rllm @ git+https://github.com/rllm-org/rllm.git"
```

this installs dependencies for running rllm cli, which uses Tinker as the training backend. 

To use `verl` as the training backend (GPU machine required), install via 

```bash
# For distributed GPU training (verl + vLLM/SGLang)
uv pip install rllm[verl] @ git+https://github.com/rllm-org/rllm.git
```

For building from source or Docker, see the [installation guide](https://docs.rllm-project.com/installation).

## Quickstart

### Option A: CLI (no code needed)

```bash
# 1. Configure your model provider
rllm model setup

# 2. Evaluate on a benchmark
rllm eval gsm8k

# 3. Train with RL
rllm train gsm8k
```

### Option B: Python API

Define a rollout (your agent) and an evaluator (your reward function), then hand them to the trainer:

```python
# my_flow.py
from openai import OpenAI
import rllm
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Trajectory

@rllm.rollout
def solve(task: Task, config: AgentConfig) -> Episode:
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": task.data["question"]}],
    )
    answer = response.choices[0].message.content or ""
    return Episode(
        trajectories=[Trajectory(name="solver", steps=[])],
        artifacts={"answer": answer},
    )
```

```python
# my_evaluator.py
import rllm
from rllm.experimental.eval.types import EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode

@rllm.evaluator
def score(task: dict, episode: Episode) -> EvalOutput:
    answer = _extract_agent_answer(episode)
    is_correct = answer.strip() == task["ground_truth"].strip()
    reward = 1.0 if is_correct else 0.0
    return EvalOutput(reward=reward, is_correct=is_correct,
                      signals=[Signal(name="accuracy", value=reward)])
```

```python
# train.py
from rllm.experimental.unified_trainer import AgentTrainer

trainer = AgentTrainer(
    backend="tinker",
    agent_flow=solve,
    evaluator=score,
    config=config,
    train_dataset=dataset,
)
trainer.train()
```

During training, `config.base_url` points to a gateway that transparently captures token IDs and logprobs — your agent code stays the same for eval and training.

See the [cookbooks](./cookbooks) for complete working examples (single-turn VLM solver, multi-agent solver-judge, and more).

## Architecture

rLLM follows a simple pipeline: **run your agent → collect traces → compute rewards → update the model**.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Your Agent  │───▶│    Traces     │───▶│   Rewards    │───▶│  RL Update   │
│  (any code)  │    │  (auto-logged)│    │ (your logic) │    │  (GRPO etc.) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

Your agent runs as-is — rLLM's SDK intercepts LLM calls and structures them into **Episodes** (one task) containing **Trajectories** (one agent run) made of **Steps** (one LLM call). A reward function scores the result, and the RL algorithm updates the model weights. The same agent code works for both eval and training.

Under the hood:
- **Workflow Engine** runs N parallel agent instances to collect rollouts
- **LiteLLM Proxy** routes requests and captures token IDs + logprobs
- **Transform Pipeline** groups trajectories for advantage computation
- **Training Backend** (verl or tinker) handles the policy update

## Community Projects

- [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) — Open-source AI researchers by Alibaba NLP [![Stars](https://img.shields.io/github/stars/Alibaba-NLP/DeepResearch)](https://github.com/Alibaba-NLP/DeepResearch)
- [Terminal-Bench-RL](https://github.com/Danau5tin/terminal-bench-rl) — Training long-horizon terminal agents with RL [![Stars](https://img.shields.io/github/stars/Danau5tin/terminal-bench-rl)](https://github.com/Danau5tin/terminal-bench-rl)
- [PettingLLMs](https://github.com/pettingllms-ai/PettingLLMs) — Multi-agent RL with on-policy training [![Stars](https://img.shields.io/github/stars/pettingllms-ai/PettingLLMs)](https://github.com/pettingllms-ai/PettingLLMs)
- [SETA](https://github.com/camel-ai/seta) — Scaling environments for terminal agents [![Stars](https://img.shields.io/github/stars/camel-ai/seta)](https://github.com/camel-ai/seta)
- [LLM-in-Sandbox](https://github.com/llm-in-sandbox/llm-in-sandbox) — Building general agents by running LLMs in a sandbox [![Stars](https://img.shields.io/github/stars/llm-in-sandbox/llm-in-sandbox)](https://github.com/llm-in-sandbox/llm-in-sandbox)
- [Cogito, Ergo Ludo](https://www.arxiv.org/abs/2509.25052) — An agent that learns to play by reasoning and planning
- [Cut the Bill, Keep the Turns](https://agate-slipper-ef0.notion.site/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c) — Affordable multi-turn search RL
- [Experiential Reinforcement Learning](https://arxiv.org/abs/2602.13949) — Experience-reflection-consolidation loop for RL with sparse rewards
- [V1: Unifying Generation and Self-Verification](https://arxiv.org/abs/2603.04304) — Pairwise self-verification for parallel test-time scaling
## Articles & Blog Posts

- [rLLM UI: Real-Time Observability Tool for Agent Training & Evaluation](https://rllm-project.com/post.html?post=rllm_ui.md) — Mar 2026
- [rLLM On-Policy Distillation: Training Smaller Students from Stronger Teachers](https://rllm-project.com/post.html?post=opd.md) — Mar 2026
- [Faster and Better: Open-Source Recipe for Deep Research Agents with Fully Async Training](https://rllm-project.com/post.html?post=async_rl.md) — Feb 2026
- [rLLM-FinQA: How a 4B Model Outperforms 235B and Rivals Gemini 2.5 Pro on Financial Analysis](https://rllm-project.com/post.html?post=finqa.md) — Feb 2026
- [rLLM SDK: Training Any Agentic Program without Code Changes](https://rllm-project.com/post.html?post=sdk.md) — Dec 2025
- [rLLM v0.2: RL Training for General Agentic Programs](https://rllm-project.com/post.html?post=rllm_v0.2.md) — Oct 2025
- [DeepSWE: Open-source SWE Agent via RL](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33) — Jul 2025
- [DeepCoder: 14B Coder at O3-mini Level](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) — Apr 2025
- [DeepScaleR: 1.5B Surpasses O1-Preview](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) — Feb 2025

## Acknowledgements

Our work is done as part of [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). The rLLM team is generously supported by grants from [Laude Institute](https://www.laude.org/), [AWS](https://aws.amazon.com/), [Hyperbolic](https://www.hyperbolic.ai/), [Fireworks AI](https://fireworks.ai/), and [Modal](https://modal.com/). We pay special thanks to [Together AI](https://www.together.ai/) for the research partnership and compute support.

## Citation

```bibtex
@misc{rllm2025,
  title={rLLM: A Framework for Post-Training Language Agents},
  author={Sijun Tan and Michael Luo and Colin Cai and Tarun Venkat and Kyle Montgomery and Aaron Hao and Tianhao Wu and Arnav Balyan and Manan Roongta and Chenguang Wang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31}},
  note={Notion Blog},
}
```

You may also cite our prior work [DeepScaleR](https://scholar.googleusercontent.com/scholar.bib?q=info:PrmBADk39GwJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOgcuI:AAZF9b8AAAAAaPCmauIfzg8Rm9ImNYDad0uPUK8&scisig=AAZF9b8AAAAAaPCmahXsNqb1jTQBw2iPfw2vm9g&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1), [DeepCoder](https://scholar.googleusercontent.com/scholar.bib?q=info:xpZNEPI6opAJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOgjM8:AAZF9b8AAAAAaPCmlM_hb3S0tzBSVrRYBZYDLWg&scisig=AAZF9b8AAAAAaPCmlG109SG8d8230AiDP4jMxlw&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1), and [DeepSWE](https://scholar.googleusercontent.com/scholar.bib?q=info:J9rT3SnY_aMJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOg3D4:AAZF9b8AAAAAaPCmxD7Nl0xA_AcAeydpcE1BXCo&scisig=AAZF9b8AAAAAaPCmxE2Spzf5lf-2Toys5xEpnuA&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1).
