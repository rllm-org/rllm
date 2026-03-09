In rLLM, we categorize agents into two types based on how they accumulate context over time:

!!! warning "Deprecation warning"
    The stepwise mode `per_step` will be soon deprecated, together with the legacy trainers (`agent_workflow_trainer` for `Verl` or `tinker_workflow_trainer` for `Tinker`). For the new [`UnifiedTrainer`](../experimental/unified-trainer.md)-based workflow trainer, the advantage calculation (via grouping) and broadcasting pattern is explicitly codified in the [run format guide](./agent-run-format.md), and will allow user-customization.

1. **Cumulative Agents**
    
    These agents accumulate their full interaction history across turns. At each step, the environment observation and the model’s response are appended to the existing prompt, forming a single long trajectory containing all previous interactions. This setting is suitable when the entire context fits within the model’s window.
    
2. **Non-Cumulative Agents**
    
    In many realistic scenarios, the full trajectory exceeds the model's context limit. For example, long observations or internal agent reasoning can bloat the prompt. In such cases, agents must **manage context** and rely on **summarized state representations**. These agents are better modeled as a **Markov Decision Process (MDP)**, where each prompt represents a compact state summarizing history, and the action is the model's response.
    

rLLM supports the following RL algorithms:

**For Cumulative Agents:**

- **GRPO with Observation Masking**
    
    The agent’s trajectory consists of concatenated tokens from all prior turns. During training, we **mask out all non model-generated tokens** (e.g. system prompts, environment observations), and compute loss only over model-generated tokens.
    
    Each full trajectory is assigned a scalar reward, and standard GRPO is applied by batching and grouping multiple trajectories over the same task to compute advantages.
    
    This method is used to train models like **DeepSWE**, and is adopted in recent work such as [ReTool](https://arxiv.org/pdf/2504.11536), [Search-R1](https://arxiv.org/pdf/2503.09516), and [RAGEN](https://arxiv.org/pdf/2504.20073). 
    

**For Non-Cumulative Agents:**

Each step is an **independent prompt-response interaction**, and a full trajectory is a **sequence of steps**. Since trajectories may vary in length (e.g., one with 10 steps, another with 8), a key challenge is **how to group steps and assign advantages** during training. rLLM supports two approaches:

- **Stepwise GRPO with Advantage Broadcasting**
    
    We first group and compute the advantage **only on the final step** using the trajectory's terminal reward. This final-step advantage is then **broadcast** to all previous steps of the trajectory.
    
    This method is suitable when earlier actions meaningfully contribute to the final outcome but don’t have fine-grained rewards.
    
- **Stepwise GRPO with Per-Step Grouping**
    
    In this approach, each step receives its **own reward**, and steps are grouped across trajectories based on their position (e.g. all step-1s, all step-2s, etc.). We then compute **step-level advantages** for each group.
    
    This formulation works best when trajectories are **symmetric**, i.e., each step is a refinement over the previous.
    
    A common example is a math or coding agent that performs **iterative self-refinement (**e.g. [Kevin-32B](https://cognition.ai/blog/kevin-32b)):
    
    - At each step, the agent outputs a solution.
    - A reward is assigned based on the correctness of that step’s solution.
    - Since each step has independent evaluability, stepwise grouping becomes natural.