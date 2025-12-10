<div align="center">

# rLLM

<div>
🚀 Reinforcement Learning for Language Agents🌟
</div>
</div>
<div>
<br>

<div align="center">
  
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://rllm-project.readthedocs.io/en/latest)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/BDH46HT9en)
[![Website](https://img.shields.io/badge/Site-%233f72af.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://rllm-project.com)
[![Blogs](https://img.shields.io/badge/Blogs-007AFF?style=for-the-badge)](https://rllm-project.com/blog)
[![X](https://img.shields.io/badge/-black?logo=X&style=for-the-badge)](https://x.com/rllm_project)

</div>

</div>

rLLM is an open-source framework for post-training language agents via reinforcement learning. With rLLM, you can easily build your custom agents and environments, train them with reinforcement learning, and deploy them for real-world workloads.

## Releases 📰

<strong>[2025/10/16]</strong> rLLM [v0.2](https://github.com/rllm-org/rllm/tree/v0.2) is now officially released! We introduce `AgentWorkflowEngine` for training over arbitrary agentic programs. It also comes integrated with the official `verl-0.5.0`, featuring support for Megatron training. Check out this [blog post](https://rllm-project.com/post.html?post=rllm_v0.2.md) for more.

<strong>[2025/07/01]</strong> We release [`DeepSWE-Preview`](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[…]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33?pvs=73), a 32B software engineering agent (SWE) trained with purely RL that achieves 59% on SWEBench-Verified with test-time scaling,(42.2% Pass@1), topping the SWEBench leaderboard for open-weight models.

<strong>[2025/04/08]</strong> We release [`DeepCoder-14B-Preview`](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), a 14B coding model that achieves an impressive **60.6%** Pass@1 accuracy on LiveCodeBench (+8% improvement), matching the performance of `o3-mini-2025-01-031 (Low)` and `o1-2024-12-17`. 

<strong>[2025/02/10]</strong> We release [`DeepScaleR-1.5B-Preview`](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), a 1.5B model that surpasses O1-Preview and achieves <strong>43.1% Pass@1</strong> on AIME. We achieve this by iteratively scaling Deepseek's GRPO algorithm from 8K→16K->24K context length for thinking.

## Getting Started 🎯

### Step 1: Building rLLM

rLLM requires Python >= 3.11. You can install it either directly via pip or build from source.

**Option A: Direct Installation**

```bash
uv pip install "git+https://github.com/rllm-org/rllm.git"
```

**Option B: Building from Source**

```bash
# Clone the repository
git clone https://github.com/rllm-org/rllm.git
cd rllm

# Create a conda environment
conda create -n rllm python=3.11 -y
conda activate rllm

# Build rLLM from source
uv pip install -e .
```

### Step 2: Installing Training Backend

rLLM supports two training backends: `verl` and `Tinker`. Choose one based on your needs.

**Option A: Using `verl` as Training Backend**

```bash
# Initialize and update the verl submodule
git submodule update --init --recursive

# Install verl
bash scripts/install_verl.sh
```

**Option B: Using `Tinker` as Training Backend**

No additional installation needed. `tinker` and `tinker-cookbook` are pre-installed when you install rLLM.

> **Note:** The direct pip installation (Option A in Step 1) comes with Tinker as the default training backend. If you want to use `verl`, you'll need to build from source (Option B in Step 1) and then install `verl` (Option A in Step 2).

### Installation with Docker 🐳

For a containerized setup, you can use Docker:

```bash
# Build the Docker image
docker build -t rllm .

# Create and start the container
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
docker start rllm-container

# Enter the container
docker exec -it rllm-container bash
```

## Awesome Projects using rLLM 🔥

* [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2): Surpassing O1-Preview with a 1.5B Model by Scaling RL
* [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51): A Fully Open-Source 14B Coder at O3-mini Level
* [DeepSWE](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[%E2%80%A6]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33): Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL
* [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch): A New Era of Open-Source AI Researchers [![GitHub Repo stars](https://img.shields.io/github/stars/Alibaba-NLP/DeepResearch)](https://github.com/Alibaba-NLP/DeepResearch)
* [Terminal-Bench-RL](https://github.com/Danau5tin/terminal-bench-rl): Training Long-Horizon Terminal Agents with Reinforcement Learning [![GitHub Repo stars](https://img.shields.io/github/stars/Danau5tin/terminal-bench-rl)](https://github.com/Danau5tin/terminal-bench-rl)
* [Cogito, Ergo Ludo](https://www.arxiv.org/abs/2509.25052): An Agent that Learns to Play by Reasoning and Planning
* [PettingLLMs](https://pettingllms-ai.github.io/): Using On-Policy Reinforcement Learning for Stronger Multi-Agent System [![GitHub Repo stars](https://img.shields.io/github/stars/pettingllms-ai/PettingLLMs)](https://github.com/pettingllms-ai/PettingLLMs)



## Acknowledgements
Our work is done as part of [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). The rLLM team is generously supported by grants from [Laude Institute](https://www.laude.org/), [AWS](https://aws.amazon.com/), [Hyperbolic](https://www.hyperbolic.ai/), [Fireworks AI](https://fireworks.ai/), and [Modal](https://modal.com/). We pay special thanks to [Together AI](https://www.together.ai/) for the research partnership and compute support. 

## Citation
```bibtex
@misc{rllm2025,
  title={rLLM: A Framework for Post-Training Language Agents},
  author={Sijun Tan and Michael Luo and Colin Cai and Tarun Venkat and Kyle Montgomery and Aaron Hao and Tianhao Wu and Arnav Balyan and Manan Roongta and Chenguang Wang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31}},
  note={Notion Blog}
  year={2025}
}
```

You may also cite our prior work [DeepScaleR](https://scholar.googleusercontent.com/scholar.bib?q=info:PrmBADk39GwJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOgcuI:AAZF9b8AAAAAaPCmauIfzg8Rm9ImNYDad0uPUK8&scisig=AAZF9b8AAAAAaPCmahXsNqb1jTQBw2iPfw2vm9g&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1), [DeepCoder](https://scholar.googleusercontent.com/scholar.bib?q=info:xpZNEPI6opAJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOgjM8:AAZF9b8AAAAAaPCmlM_hb3S0tzBSVrRYBZYDLWg&scisig=AAZF9b8AAAAAaPCmlG109SG8d8230AiDP4jMxlw&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1), and [DeepSWE](https://scholar.googleusercontent.com/scholar.bib?q=info:J9rT3SnY_aMJ:scholar.google.com/&output=citation&scisdr=CgIJFx-xEMCQ6zOg3D4:AAZF9b8AAAAAaPCmxD7Nl0xA_AcAeydpcE1BXCo&scisig=AAZF9b8AAAAAaPCmxE2Spzf5lf-2Toys5xEpnuA&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1).
