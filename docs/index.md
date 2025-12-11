# rLLM: Reinforcement Learning for Language Agents

rLLM is a framework for training language agents with reinforcement learning. It lets you define custom agents and environments, collect trajectories, and run scalable RL training loops to continuously improve your agents' performance.

## Key Features

rLLM provides:

- **Simple abstractions for building and training custom agents**: rLLM cleanly separates agent and environment design from the underlying training infrastructure. You focus on defining agents and environments; rLLM handles the training details.

- **Unified interface for agent inference and training**: Training and deploying LLM agents traditionally requires two separate stacks for serving and training. rLLM provides a single interface for both, making it easy to continuously evolve agents that "learn from experience."

- **Efficient trajectory generation and scalable RL training**: rLLM's execution engine supports asynchronous, parallelized trajectory generation and large-scale RL optimization.

## What's New in v0.2.x

- **rLLM SDK (preview):** The rLLM SDK enables you to transform agents written in frameworks such as LangGraph, SmolAgent, or Strands into trainable workflows. Check out this [LangGraph RAG example](examples/sdk_langgraph_rag.md), which builds a RAG agent and trains it with the rLLM SDK.

- **Tinker training backend:** In addition to `verl`, rLLM now supports `Tinker` as a training backend. You can use the same abstractions for building agents and easily switch between different backends for training. 

- **Multi-agent training:** rLLM now supports multi-agent training. Check out our [Solver–Judge workflow](examples/solver_judge.md) to see how you can jointly optimize a solver and judge agent with RL.

- **VLM training:** rLLM supports Vision-Language Model training with the `verl` backend. See the [Geo3K training example](examples/vlm.md) for reference.

- **LoRA fine-tuning:** rLLM supports LoRA training in both the `verl` and `Tinker` backends. See the [GSM8K LoRA example](examples/gsm8k_lora.md) for how to enable LoRA training with a single config change.

- **Eval Protocol Integration** We integrate with the [Eval Protocol](https://evalprotocol.io/) from Fireworks AI. Users can now train on any environments supported by the Eval Protocol. See this [example](examples/eval_protocol_frozen_lake.md) that uses Eval Protocol in rLLM to train a Frozenlake agent.

## Getting Started

To get started with rLLM, see the [Installation Guide](getting-started/installation.md) and [Quick Start tutorial](getting-started/quick-start.md).

## Examples and Tutorials

rLLM is designed to be extensible. You can easily build and train custom agents and environments using our modular API and training engine. Walk through the [core concepts](core-concepts/overview.md) and browse the [examples on GitHub](https://github.com/rllm-org/rllm/tree/main/examples) to understand the fundamentals of rLLM and adapt them to your own use cases.

## Community & Support

rLLM is an open-source project under active development. We welcome contributions, bug reports, and feature requests from the community.
Please read our [Contributing guide](contributing.md) before contributing to rLLM.

