# Examples

This section contains examples demonstrating how to use rLLM to train agents for various tasks.

## Available Examples

### 🧩 [rLLM SDK](../core-concepts/sdk.md)
Train agents using the rLLM SDK, including tutorials for a math agent, solver-judge workflow, and LangGraph RAG agent.

### 🚀 [RL Training with Tinker](tinker_rl.md)
Train a solver‑judge RL workflow using Tinker's hosted GPU service.

### 💡 [LoRA Training with Verl](gsm8k_lora.md)
Fine-tune a math reasoning agent on GSM8K with LoRA using `verl` as training backend.

### ⚖️ [Solver-Judge Workflow](solver_judge.md)
Train a multi-agent workflow to sample multiple candidate solutions, then use a judge to select the best.

### 👁️ [Vision-Language Models (VLM)](vlm.md)
Train multimodal agents that can reason over both images and text, demonstrated with geometry problem solving.

### 🧮 [DeepScaler](deepscaler.md) & 💻 [DeepCoder](deepcoder.md)
Train reasoning models that aces math competition (e.g. DeepScaleR) and coding contests (e.g. DeepCoder)

### 🛠️ [DeepSWE](swe.md)
Train an autonomous SWEAgent that can write software patches to resolve real-world Github issues.

### 🔍 [Search Agent](search.md) 
Build agents that can search and retrieve information effectively.

### 🎮 [Frozenlake Agent](frozenlake.md)
Classic RL examples using environments like FrozenLake.

### 🧊 [Eval Protocol Integration (FrozenLake)](eval_protocol_frozen_lake.md)
Use Eval Protocol benchmarks as rLLM workflows for evaluation and RL training.

### 📚 [Math SFT Training](sft.md)
Supervised fine-tuning of base math models(e.g. Qwen/Qwen2.5-Math-1.5B) using high-quality trajectories generated from teacher models (e.g. DeepScaleR)