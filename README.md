<div align="center">

# rLLM

<div>
🚀 Reinforcement Learning for Language Agents🌟
</div>
</div>
<div>
<br>

<div align="center">
  
[![Documentation](https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=googledocs&logoColor=white)](https://rllm-project.readthedocs.io/en/latest)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/BDH46HT9en)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://www.agentica-project.com) 
[![Blog](https://img.shields.io/badge/Blog-007AFF?style=for-the-badge)](https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31)
[![Hugging Face Collection](https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/agentica-org)

</div>

</div>

rLLM is an open-source framework for post-training language agents via reinforcement learning. With rLLM, you can easily build your custom agents and environments, train them with reinforcement learning, and deploy them for real-world workloads.

## Releases 📰

<strong>[2025/09/16]</strong> rLLM [v0.2](https://github.com/rllm-org/rllm/tree/v0.2) is now in development and available in preview. It comes integrated with verl-0.5.0, featuring support for Megatron training. Stay tuned for more updates!

<strong>[2025/07/01]</strong> We release [`DeepSWE-Preview`](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[…]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33?pvs=73), a 32B software engineering agent (SWE) trained with purely RL that achieves 59% on SWEBench-Verified with test-time scaling,(42.2% Pass@1), topping the SWEBench leaderboard for open-weight models.

- 🍽️ An In-Depth Blog Post on our [SWE Agents and RL Training Recipes](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[…]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33?pvs=73)
- 🤗 HF Model [`DeepSWE-Preview`](https://huggingface.co/agentica-org/DeepSWE-Preview)
- 🤗 HF Dataset [`R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset)
- 📄 [Training Scripts](https://github.com/rllm-org/rllm/tree/main/examples/swe)
- 📈 [Wandb Training Logs](https://wandb.ai/mluo/deepswe)—All training runs and ablations.
- 🔎 [Evaluation Logs](https://drive.google.com/file/d/10LIwpJeaFuiX6Y-qEG2a4a335PEuQJeS/view?usp=sharing)—16 passes over SWE-Bench-Verified.

<strong>[2025/04/08]</strong> We release [`DeepCoder-14B-Preview`](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), a 14B coding model that achieves an impressive **60.6%** Pass@1 accuracy on LiveCodeBench (+8% improvement), matching the performance of `o3-mini-2025-01-031 (Low)` and `o1-2024-12-17`. 

<strong>[2025/02/10]</strong> We release [`DeepScaleR-1.5B-Preview`](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), a 1.5B model that surpasses O1-Preview and achieves <strong>43.1% Pass@1</strong> on AIME. We achieve this by iteratively scaling Deepseek's GRPO algorithm from 8K→16K->24K context length for thinking.

## Getting Started 🎯

### Installation

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/rllm-org/rllm.git
cd rllm

# create a conda environment
conda create -n rllm python=3.10 (use python=3.11 for MacOS)
conda activate rllm

# Install all dependencies
pip install -e ./verl
pip install -e .

**Note:** On macOS, GPU features (flash-attn, deepspeed, vllm) are automatically excluded for compatibility. For GPU support on macOS, you can install with: `pip install -e .[gpu]`.
```

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

* [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch): A New Era of Open-Source AI Researchers [![GitHub Repo stars](https://img.shields.io/github/stars/Alibaba-NLP/DeepResearch)](https://github.com/Alibaba-NLP/DeepResearch)
* [Terminal-Bench-RL](https://github.com/Danau5tin/terminal-bench-rl): Training Long-Horizon Terminal Agents with Reinforcement Learning [![GitHub Repo stars](https://img.shields.io/github/stars/Danau5tin/terminal-bench-rl)](https://github.com/Danau5tin/terminal-bench-rl)
* [Cogito, Ergo Ludo](https://www.arxiv.org/abs/2509.25052): An Agent that Learns to Play by Reasoning and Planning


## Acknowledgements

- Our training experiments are powered by our heavily modified fork of [verl](https://github.com/volcengine/verl), an open-source RLHF library.
- Our models are trained on top of [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [`DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), and [`Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32b).
- Our work is done as part of [Berkeley Sky Computing Lab](https://skycomputing.berkeley.edu/), [Berkeley AI Research](https://bair.berkeley.edu/), and a successful collaboration with Together AI.


## Citation

Citing rLLM:

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

Citing DeepSWE:

```bibtex
@misc{deepswe2025,
  title={DeepSWE: Training a State-of-the-Art Coding Agent from Scratch by Scaling RL},
  author={Michael Luo and Naman Jain and Jaskirat Singh and Sijun Tan and Ameen Patel and Qingyang Wu and Alpay Ariyak and Colin Cai and Tarun Venkat and Shang Zhu and Ben Athiwaratkun and Manan Roongta and Ce Zhang and Li Erran Li and Raluca Ada Popa and Koushik Sen and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33}},
  note={Notion Blog},
  year={2025}
}
```

Citing DeepCoder:

```bibtex
@misc{deepcoder2025,
  title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
  author={Michael Luo and Sijun Tan and Roy Huang and Ameen Patel and Alpay Ariyak and Qingyang Wu and Xiaoxiang Shi and Rachel Xin and Colin Cai and Maurice Weber and Ce Zhang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
  note={Notion Blog},
  year={2025}
}
```

Citing DeepScaleR:

```bibtex
@misc{deepscaler2025,
  title={DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL},
  author={Michael Luo and Sijun Tan and Justin Wong and Xiaoxiang Shi and William Y. Tang and Manan Roongta and Colin Cai and Jeffrey Luo and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2}},
  note={Notion Blog}
  year={2025}
}
```
