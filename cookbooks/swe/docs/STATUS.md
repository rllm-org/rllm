# SWE Agent RL Training in rLLM: Status Report

## 1. Project Goal: SWE-Agent RL Training in rLLM

The goal of this project was to train a strong software engineering agent using verl and rLLM. The final system lives in [`cookbooks/swe`](https://github.com/rllm-org/rllm/tree/swe/cookbooks/swe) and covers the full lifecycle needed for SWE-agent training: preparing datasets, launching agent rollouts, executing tool calls in isolated sandboxes, grading patches, capturing trajectories, transforming multi-turn rollouts into trainable records, running GRPO training, and validating on held-out SWE benchmarks. The central engineering challenge was that SWE tasks are long-horizon, stateful, and easy to contaminate. A model must inspect a repository, edit files, run tests, and submit a patch, while the training system must preserve the actual model actions, hide information that should not be visible to the agent, and grade the final result in a controlled environment. This required work across the agent loop, the evaluator, the model gateway, the training data transform, and the launch infrastructure. The final contribution is a pipeline for SWE RL and a systems integration that made this workload practical inside rLLM.

## 2. Overall Architecture

The final system is organized as a rollout-to-training pipeline. A task is loaded from rLLM's dataset layer, passed into [`SWEAgentFlow`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/agent_flow.py), executed by mini-swe-agent inside a Modal/SWE-ReX sandbox, converted into an rLLM `Episode`, graded by [`SWEEvaluator`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/evaluator.py), and transformed into training data for veRL. During evaluation, the model wrapper can call any OpenAI-compatible endpoint. During training, the same wrapper points at the rLLM model gateway, which captures token IDs, logprobs, and traces without changing the agent loop.

```text
DatasetRegistry task
  -> SWEAgentFlow
  -> mini-swe-agent DefaultAgent
  -> OpenAIClientModel
  -> model endpoint or rLLM gateway
  -> Modal / SWE-ReX sandbox
  -> submitted patch
  -> SWEEvaluator
  -> reward, correctness, diagnostics
  -> trajectory transform
  -> veRL / GRPO training update
```

The important design choice is that the agent code is shared between evaluation and training. The same mini-swe-agent loop issues bash tool calls, receives observations, edits the repository, and submits a patch in both modes. The difference is the model endpoint and the trace-capture path: evaluation can use a normal external model provider, while training routes model calls through the rLLM gateway so the trainer can reconstruct the policy actions with token-level metadata. This keeps the system modular while reducing train/eval drift.

The architecture also separates responsibilities cleanly. Dataset-specific modules prepare repositories and define grading behavior. The agent flow manages rollout limits, context compaction, sandbox lifecycle, and patch generation. The evaluator owns reward computation and diagnostics. The gateway owns token/logprob capture. The veRL transform owns conversion from multi-turn agent traces into trainable tensors. This separation made it possible to improve each part independently while still producing a single end-to-end SWE RL pipeline.

## 3. Unified SWE Task Interface

The cookbook supports multiple SWE task families behind one shared task and evaluator interface. The supported task families are SWE-smith, SWE-rebench V2, SWE-bench Multilingual, and SWE-bench Pro. Each dataset has different metadata, repository setup requirements, grading logic, and hidden-test behavior, but the rollout and training loop should not need to know those details. The task interface provides the common fields needed by the agent, such as the repository, problem statement, patch metadata, and grading configuration, while task-specific modules handle dataset-specific setup and evaluation. This abstraction made it possible to use different datasets for different stages of training and validation. SWE-smith and SWE-rebench V2 were used as training data because they provide a large number of tasks and support controlled curation. SWE-bench Multilingual and SWE-bench Pro were used as validation datasets because they better measure transfer to more realistic or more difficult held-out tasks. Keeping these task families behind a common interface also made it easier to compare validation metrics across datasets and to reuse the same agent flow, sandbox policy, and evaluator structure.

## 4. mini-swe-agent Integration

The SWE cookbook uses mini-swe-agent as the harness for our software engineering agent, with the local bootstrap handled by [`environment.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/environment.py). This was important because the project did not need to invent a new repository-editing agent from scratch. Instead, it adapted a compact existing SWE agent into the rLLM rollout interface. [`SWEAgentFlow`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/agent_flow.py) creates the task environment, instantiates mini-swe-agent's `DefaultAgent`, gives it the problem statement, lets it issue bash tool calls, and converts the final submission into an rLLM `Episode`.

The integration also required a custom model wrapper. [`OpenAIClientModel`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/openai_model.py) implements the mini-swe-agent model protocol while talking to any OpenAI-compatible endpoint. This lets the same agent run against external evaluation models, hosted vLLM models, or the rLLM model gateway during training. The wrapper uses native tool calling for the bash tool, handles malformed tool-call output by returning format feedback to the agent, supports optional context summarization, and captures token-aware response metadata when the backend exposes token IDs.

This design kept evaluation and training aligned. In evaluation mode, `config.base_url` can point directly at an OpenAI-compatible model endpoint. In training mode, the same mini-swe-agent loop points at the rLLM gateway, which transparently records model calls, token IDs, logprobs, and traces for RL. The agent code therefore stays almost identical across offline evaluation and RL rollout collection, which reduces train/eval mismatch and makes the cookbook easier to maintain.

## 5. Modal and SWE-ReX Sandbox Execution

The agent interacts with repositories through Modal sandboxes managed by SWE-ReX. This is a central part of the system because SWE tasks require real command execution: the model must inspect files, run tests, apply edits, and produce a patch inside a live repository checkout. [`environment.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/environment.py) loads the mini-swe-agent runtime configuration, creates a `swerex_modal` environment from the task's Docker image and working directory, and applies the runtime patches needed for reliable Modal execution.

The Modal/SWE-ReX layer also carries much of the practical engineering needed for large-scale training. Sandbox creation can fail transiently, Modal tunnels can drop, image startup can stall, and long runs can leak remote sessions if cleanup is not strict. The cookbook adds retry logic around sandbox creation in [`SWEAgentFlow`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/agent_flow.py), command-level retry helpers in [`tasks/common.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/tasks/common.py), and SWE-ReX compatibility patches for Modal deployment and remote runtime behavior. These patches handle GCR/GAR images, bound Modal control-plane waits, fix sandbox termination, adapt mini-swe-agent v2 protocol behavior, and retry transient Modal tunnel failures.

The rollout path also includes explicit load-smoothing to make large batches robust to Modal rate limits and control-plane pressure. [`SWEAgentFlowConfig`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/flow_config.py) exposes `startup_jitter_s`, which adds a random sleep at the start of each rollout so many workers do not try to create Modal sandboxes at the exact same time. Sandbox creation is then retried on transient failures with randomized backoff; the default policy uses 3 attempts with 5-10 seconds of backoff between attempts. After a sandbox is running, command execution uses retry helpers for known transient Modal and package-install failures, with exponential backoff at the command level. These mechanisms are small individually, but they are important for training because a large RL batch can otherwise fail due to infrastructure bursts rather than model behavior.

This Modal execution layer is what made the SWE workload realistic. The model was not answering static prompts; it was acting inside real repository environments with shell access, command timeouts, startup timeouts, and dataset-specific setup. It also made the clean grading design possible, because the system could create one sanitized sandbox for agent interaction and a separate fresh sandbox for grading. For the final report, this should be treated as a first-class contribution: reliable remote sandbox orchestration was necessary for turning SWE tasks into scalable RL episodes.

## 6. Dataset Curation: SWE-Smith Filtering and SWE-Rebench V2 Patch-Length Analysis

Dataset curation was a major part of the project because raw SWE datasets contain many tasks that are either too easy, too hard, or not useful for RL. For policy-gradient training, the most useful tasks are those where the model sometimes succeeds and sometimes fails. If a task always fails, it contributes little reward signal and wastes rollout budget. If a task always succeeds, the reward is constant and the training signal is weak. The SWE-smith curation therefore focused on the middle band, defined as `0 < success_rate < 1`, using trajectories generated by `Qwen3.5-35B-A3B`.

The five SWE-smith trajectory datasets used for curation were:

| Language | Dataset |
|---|---|
| Python | `JWei05/swe_smith_py_qwen3.5_35b_trajs_1952` |
| Rust | `JWei05/swe_smith_rs_qwen3.5_35b_trajs_2477` |
| Go | `JWei05/swe_smith_go_qwen3.5_35b_trajs_1448` |
| JavaScript | `JWei05/swe_smith_js_qwen3.5_35b_trajs_4358` |
| Java | `JWei05/swe_smith_java_qwen3.5_35b_trajs_4369` |

Across these five datasets there were 18,629 total rows. For analysis, I treated rows with `n_runs_completed` between 3 and 5 as the reliable subset, because success rates computed from too few completed rollouts are unstable. This left 10,185 reliable rows. The success-rate distribution was strongly bimodal: 39.0% of reliable rows had `success_rate = 0`, 33.2% had `success_rate = 1`, and only 27.8% fell into the useful middle band `0 < success_rate < 1`.

| Language | Total rows | Reliable rows, 3-5 runs | `success_rate = 0` | `0 < success_rate < 1` | `success_rate = 1` |
|---|---:|---:|---:|---:|---:|
| Python | 2,000 | 1,394 | 712, 51.1% | 288, 20.7% | 394, 28.3% |
| Rust | 5,000 | 1,295 | 663, 51.2% | 398, 30.7% | 234, 18.1% |
| Go | 1,629 | 1,177 | 329, 28.0% | 444, 37.7% | 404, 34.3% |
| JavaScript | 5,000 | 2,820 | 863, 30.6% | 894, 31.7% | 1,063, 37.7% |
| Java | 5,000 | 3,499 | 1,405, 40.2% | 809, 23.1% | 1,285, 36.7% |
| **Total** | **18,629** | **10,185** | **3,972, 39.0%** | **2,833, 27.8%** | **3,380, 33.2%** |

The final SWE-smith sampling recipe used the `0 < success_rate < 1` filter and sampled up to 300 rows per language. Before sampling, the raw middle-band candidate counts were 315 Python rows, 443 Rust rows, 474 Go rows, 1,036 JavaScript rows, and 881 Java rows. This produced a balanced 1,500-row SWE-smith component with 300 rows from each language.

The older curation analysis also identified broken-image repositories, where all instances in a repository had zero success because the environment was broken rather than because the problems were genuinely hard. The recommended broken-image drop list removed 706 rows from the reliable SWE-smith corpus, with Java accounting for most of those failures. This was important because leaving broken environments in the data would bias the dataset toward impossible tasks and make `success_rate = 0` less meaningful as a difficulty signal.

Patch length was the most useful patch-shape feature in SWE-smith, but its usefulness varied by language. The analysis found that patch length was strongly predictive for Python, with AUC 0.82 for predicting whether the model could solve a task at all. It was moderately useful for JavaScript and Go, with AUC 0.70 and 0.69. It was much weaker for Rust and Java, with AUC 0.60 and 0.57, which means patch length alone was not enough to build a good difficulty filter for those languages. This shaped the final strategy: use success-rate filtering for SWE-smith, and use patch length more heavily for SWE-rebench V2 where patch-size variation better matched the benchmark difficulty distribution.

For SWE-rebench V2, I analyzed whether patch length correlated with task difficulty using `gpt-5-mini` and `gpt-5.4-nano`. The sweep sampled up to 20 instances per language and patch-length bucket across eight languages: Python, Go, PHP, Java, JavaScript, TypeScript, C, and C++. Patch length was measured as `log10(max(golden_patch_line_count, 1))`, and examples were grouped into four buckets: `[0.0, 1.5]`, `[1.5, 2.0]`, `[2.0, 2.5]`, and `[2.5, inf)`.

Overall, `gpt-5-mini` solved 113 out of 640 selected tasks, for 17.7% accuracy, while `gpt-5.4-nano` solved 72 out of 640, for 11.2% accuracy. The bucketed results showed a clear relationship between patch length and difficulty. For `gpt-5-mini`, accuracy dropped from 30.6% in the shortest patch bucket to 18.8% in the next bucket and 10.6% in the two longest buckets. For `gpt-5.4-nano`, accuracy dropped from 18.1% to 14.4%, then to 5.6% and 6.9%. This supported the decision to emphasize easier and medium-difficulty Rebench examples rather than sampling uniformly from the entire dataset.

| Model | `[0.0, 1.5]` | `[1.5, 2.0]` | `[2.0, 2.5]` | `[2.5, inf)` | Overall |
|---|---:|---:|---:|---:|---:|
| `gpt-5-mini` | 30.6% | 18.8% | 10.6% | 10.6% | 17.7% |
| `gpt-5.4-nano` | 18.1% | 14.4% | 5.6% | 6.9% | 11.2% |

The language-level results showed that difficulty was also language-dependent. PHP was the easiest in this sweep, with 38.8% accuracy for `gpt-5-mini` and 32.5% for `gpt-5.4-nano`. Python, Go, JavaScript, and TypeScript were in the middle. C and C++ were much harder, with accuracies near 6% or below. This influenced the final mix: Python, Go, JavaScript, and TypeScript were sampled mostly from the shortest patch bucket; PHP included both the shortest and next-shortest buckets because it remained relatively learnable; Java used all shortest-bucket examples plus enough next-bucket examples to reach the target count; and C/C++ were included despite being hard because they improved language coverage.

The final mixed dataset was `JWei05/swe_smith_rebenchv2_5136`. It contained 1,500 SWE-smith rows and 3,636 SWE-rebench V2 rows. The Rebench portion selected 300 Python rows, 300 Go rows, 600 PHP rows, 300 Java rows, 300 JavaScript rows, 600 TypeScript rows, all 230 C rows repeated three times for 690 rows, and all 182 C++ rows repeated three times for 546 rows. This produced a dataset that was large enough for training, balanced across synthetic and real tasks, focused on learnable difficulty bands, and still multilingual.

## 7. Agent Context Compaction and Summarization

Agent context compaction follows the same high-level motivation described in Cursor's self-summarization writeup: long coding-agent trajectories can grow faster than the model context window, so the harness needs a way to condense earlier interaction history and continue the task instead of stopping or blindly dropping old context ([Cursor, "Training Composer for longer horizons"](https://cursor.com/blog/self-summarization)). In our SWE setting, the agent may inspect many files, run tests repeatedly, observe failures, edit code, and rerun commands before it is ready to submit a patch. Without compaction, these long episodes either exceed the prompt limit or spend most of the available context on stale observations.

The compaction loop is triggered when the prompt reaches a configured token threshold. At that point, the model is asked to summarize the older conversation state, and the agent continues with a compacted history rather than the full transcript. The key difference in our implementation is that we do not replace the entire history with only a summary. We preserve the system prompt, the original task, and the last one or two complete recent turns, depending on configuration, so the agent still has local coherence around the most recent command, observation, and plan. Older turns are summarized, while recent tool-call and tool-response pairs remain verbatim.

This is fundamentally different from training-time trajectory collapse. Context compaction changes what the agent sees during future rollout turns. Its purpose is to keep the interactive problem-solving loop alive under context limits. The compaction process records summary-related metadata, including the number of summaries used, so that long-context behavior can be monitored during evaluation and training. This made compaction an agent reliability feature: it allowed longer SWE episodes to complete without simply truncating away the information needed to continue solving the task.

## 8. Multi-Turn Trajectory Collapse for Training

Multi-turn trajectory collapse is a training-time transformation. Before this work, a multi-turn agent trajectory could be treated as many separate training instances, one per assistant step. That approach is inefficient for long SWE rollouts because each episode can produce many rows, increasing training time and causing long trajectories to receive disproportionate weight. It also makes it harder to reason about an entire agent episode as one coherent training example.

The collapse logic in [`transform.py`](https://github.com/rllm-org/rllm/blob/swe/rllm/experimental/verl/transform.py) converts a compatible multi-turn trajectory into a single training row. The response sequence is constructed from the model actions and the interleaved observations, but the loss mask is only enabled on the model-generated action tokens. Tool outputs, environment observations, and other non-action spans are included as context but masked out of the policy loss. This preserves the tokens the model should learn from while avoiding loss on text that came from the environment.

This change is important because SWE rollouts are much longer than typical single-turn RL prompts. Collapsing trajectories dramatically reduces the number of training rows produced by long episodes, improves training efficiency, and keeps the optimization target aligned with what the model actually controls. This was one of the main rLLM training contributions and is separate from rollout summarization.

## 9. Clean Sandboxing and Anti Reward-Hacking

Sandbox separation was another core part of making SWE RL reliable. In software-engineering benchmarks, the agent should not be able to inspect hidden tests, benchmark metadata, grading files, or git history that reveals the target patch. If rollout and grading happen in the same environment without care, the agent may learn shortcuts that exploit benchmark artifacts rather than learning to solve the task.

The cookbook mitigates this by using purposeful separate sandboxes for agent interaction and grading. The rollout sandbox is prepared for the agent, then git history is removed and the repository is reinitialized so the model cannot inspect previous commits to recover the answer. The agent interacts only with this sanitized environment and produces a patch. After rollout, that patch is collected and applied in a separate fresh grading sandbox. Hidden tests and grading artifacts are introduced only in the grading environment, after the agent has already submitted its patch.

This design reduces reward hacking risk in two ways. First, the agent cannot recover the solution by reading git history or benchmark internals during rollout. Second, the grading environment starts from a clean copy, so the reward is based only on whether the submitted patch applies and passes the hidden tests. This is especially important for SWE-rebench V2 and SWE-bench-style datasets, where the correctness signal depends on hidden or benchmark-specific tests.

## 10. Mitigating Train-Inference Drift

A major issue in multi-turn agent RL is train-inference drift caused by re-tokenization. During rollout, the model generates token IDs. If the training pipeline later reconstructs assistant messages as text and re-tokenizes them, the resulting IDs may differ from the IDs the model actually sampled. This creates off-policy updates because training is no longer using the exact sequence produced during rollout.

There are several common sources of this drift. First, tokenization is not unique: two different token sequences can decode to the same text and then re-encode differently. Second, tool-call serialization can change whitespace, field ordering, or formatting when a parser reads a model response and renders it back into chat messages. Third, chat templates can differ across inference, gateway, and training frameworks. A particularly important failure mode for SWE agents is malformed assistant output: if a model forgets an end `</think>` token or emits a tool call in a slightly invalid format, a chat parser may drop thinking content or tool-call content even though the model actually generated those tokens.

The mitigation was to capture model response token IDs and logprobs from the rollout path and carry them into training. The OpenAI-compatible model wrapper and model gateway were updated so vLLM token IDs can be returned and stored. The training side can then use the sampled token IDs and logprobs instead of depending only on reconstructed assistant text. This removes most drift caused by tool-call parsing, chat-template differences, and malformed response rendering.

The remaining drift is mostly the unavoidable non-unique-tokenization case when text must be decoded and re-encoded for agent-side parsing. Even there, the important improvement is that the policy update is much closer to the actual rollout sequence. This is an on-policy correctness improvement for agent RL, not merely a tokenizer cleanup.

## 11. Rich Evaluator Signals and Rollout Diagnostics

The evaluator was extended to track richer signals than a single reward number. For SWE-agent training, a failed episode can mean many different things: the model submitted an incorrect patch, the patch did not apply, the sandbox failed to build, the task timed out, the model hit the maximum turn limit, or the grading command failed for infrastructure reasons. Treating all of these as the same failure makes debugging training runs much harder.

The current system tracks termination reasons such as successful submission, timeout, max-turns exceeded, and error. It also separates invalid patches from fatal grading or setup failures where possible. For benchmark-style tasks, the evaluator can expose F2P/P2P-style signals, which help distinguish whether the model fixed failing tests, preserved passing tests, or broke existing behavior. These diagnostics are useful both during validation and during training because they show whether reward changes are coming from better patches, different episode lengths, fewer infrastructure failures, or changed agent behavior.

This observability was especially useful for the Rebench patch-length sweep. The sweep results included not only correctness but also termination counts. For example, in the 640-task sweep, `gpt-5-mini` had 597 `env_done` terminations, 30 timeouts, 11 errors, and 2 max-turns terminations. `gpt-5.4-nano` had 613 `env_done` terminations, 4 timeouts, 11 errors, and 12 max-turns terminations. These numbers made it possible to tell that the main difficulty trend was not just missing completions; most tasks completed, but correctness declined as patch length increased.

## 12. rLLM Systems Work and Production Training Infrastructure

In addition to the SWE cookbook itself, the project included general rLLM maintenance and infrastructure work needed to make long SWE training runs practical. This included making inference usable without requiring the full Verl installation, fixing workflow bookkeeping so completed episodes keep the correct task metadata, updating rLLM for compatibility with breaking changes in the Verl 0.7.1 training stack, and improving gateway cleanup so long training jobs do not accumulate stale session state.

The training path also required substantial launch and runtime work. The cookbook includes launchers for Qwen3.5 and Qwen3.6 training, Megatron-based training paths, Arnold/B200 launch flows, external Ray support, runtime defaults, and checkpoint upload support. These pieces are not the core research idea, but they are necessary for turning the SWE RL pipeline into something that can run at scale on real cluster infrastructure.

This systems work shows that the contribution was not only algorithmic. The project connected the agent environment, the model gateway, the trajectory transform, the trainer backend, and the cluster launch path into one runnable system. That integration is what made the dataset curation and training experiments possible.

## 13. Training Setup Limitations

The final status of the project is that the SWE codebase and core training pipeline are functioning, but stable end-to-end training remains blocked mainly by compute orchestration and training-environment constraints. The agent loop, dataset preparation, sandbox interaction, evaluator, trajectory capture, trajectory transform, and training launch code were brought up and validated in pieces. The limiting factor was getting all infrastructure layers to work together reliably for long full-scale runs. The SWE workload depended on the ModelChef environment, rLLM, veRL, Megatron, vLLM, Ray, Arnold, B200 workers, Modal/SWE-ReX sandboxes, HDFS artifacts, W&B logging, and checkpoint export all being correct at the same time. A failure in any one layer could prevent the run from reaching the first optimizer step, even when the agent code and dataset logic were correct.

The biggest structural issue was network topology. The B200 GPU nodes were suitable for model compute, but they were not suitable as the internet-facing driver for SWE training. In particular, B200 head nodes could not reliably reach `api.modal.com`, which made GPU-head topologies fail Modal preflight before training. They also were not the right place to own internet-facing model upload paths. This forced the training architecture into a split-role design: a CPU-only head handled Modal/SWE-ReX, gateway coordination, logging, artifact/checkpoint upload, and other networked control-plane work, while B200 nodes joined as GPU workers for model initialization, rollout serving, and training.

The CPU side had its own constraint. CPUs generally had internet connectivity, but only a small fraction had Modal/SWE-ReX connectivity good enough for high-concurrency SWE rollout. Some CPU candidates could start Arnold jobs but failed static Modal checks with no HTTP response, and others passed small probes but later showed repeated `modal.host` connection failures under full rollout load. As a result, the launcher needed CPU-driver preflights, dynamic Modal probes, and sometimes relaunches onto a better CPU host before attaching expensive B200 workers. The practical rule became: first find a Modal-good CPU driver, then attach B200 workers to that CPU Ray head.

The hardware and runtime setup was also fragile. B200 workers needed the correct CUDA driver library rather than the CUDA compatibility shim, a complete upgraded Megatron package for Qwen3.5 CP2 support, remove-padding enabled for context parallelism, and safe micro-batch defaults to avoid actor-update OOMs. The B200 topology mattered as well: cross-prefix B200 placement could reach rollout but fail during actor update with RDMA/NCCL timeouts, so durable runs needed same-minipod or same-prefix placement. For 35B A3B runs, rollout tensor parallelism introduced a separate vLLM constraint: TP=2 rollouts could not use the UniProc executor because only one of the two TP ranks joined, causing vLLM EngineCore startup to time out.

Checkpointing and artifact management also imposed constraints. Full optimizer checkpoints were too expensive for the desired runs, so the training path had to prove model-plus-extra checkpoint saves without optimizer state. HDFS mounts and artifact paths had to be writable and consistent across Arnold jobs, and Qwen3.5 checkpoint export needed to preserve original MTP tensors even when MTP was disabled for training. Several runs were blocked or restarted not because the model failed, but because checkpoint directories, HDFS provenance, save contents, or runtime artifacts were mismatched.

## 14. GPU Usage

The project used approximately 25,000 A100-equivalent GPU-hours across model serving, SWE rollouts, validation sweeps, debugging runs, and training experiments. Using the B200 training setup as an 8-GPU node and a rough `1 B200 ~= 4 A100` conversion, this corresponds to about 781 B200 node-hours, or about 33 B200 node-days.

## Implementation References

- SWE agent rollout and context compaction: [`cookbooks/swe/swe/agent_flow.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/agent_flow.py)
- SWE agent flow configuration, startup jitter, and sandbox retry policy: [`cookbooks/swe/swe/flow_config.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/flow_config.py)
- SWE runtime bootstrap and Modal environment creation: [`cookbooks/swe/swe/environment.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/environment.py)
- SWE evaluator and task routing: [`cookbooks/swe/swe/evaluator.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/evaluator.py)
- Sandbox setup, git reinitialization, and fresh grading helpers: [`cookbooks/swe/swe/tasks/common.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/tasks/common.py)
- B200 Megatron training launcher: [`cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh)
- 35B-A3B Megatron training launcher: [`cookbooks/swe/swe/training_scripts/run_swe_training_35b_a3b_megatron.sh`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/training_scripts/run_swe_training_35b_a3b_megatron.sh)
- Modal deployment compatibility patch: [`cookbooks/swe/swe/patches/swerex_modal_minimal.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/patches/swerex_modal_minimal.py)
- mini-swe-agent v2 / SWE-ReX compatibility patch: [`cookbooks/swe/swe/patches/swerex_modal_compat.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/patches/swerex_modal_compat.py)
- SWE-ReX remote retry patch: [`cookbooks/swe/swe/patches/swerex_remote_retry.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/patches/swerex_remote_retry.py)
- SWE-smith and SWE-rebench V2 mixed dataset preparation: [`cookbooks/swe/swe/scripts/prepare_swe_smith_rebenchv2_mix.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/scripts/prepare_swe_smith_rebenchv2_mix.py)
- SWE evaluation runner: [`cookbooks/swe/swe/scripts/run_eval.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/scripts/run_eval.py)
- Multi-run evaluation and success-rate collection: [`cookbooks/swe/swe/scripts/run_n_eval.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/scripts/run_n_eval.py)
- Multi-turn trajectory collapse: [`rllm/experimental/verl/transform.py`](https://github.com/rllm-org/rllm/blob/swe/rllm/experimental/verl/transform.py)
- Token-aware OpenAI-compatible model wrapper: [`cookbooks/swe/swe/openai_model.py`](https://github.com/rllm-org/rllm/blob/swe/cookbooks/swe/swe/openai_model.py)
- Gateway token/logprob extraction: [`rllm-model-gateway/src/rllm_model_gateway/data_process.py`](https://github.com/rllm-org/rllm/blob/swe/rllm-model-gateway/src/rllm_model_gateway/data_process.py)
- Conceptual background for context compaction and self-summarization: [Cursor, "Training Composer for longer horizons"](https://cursor.com/blog/self-summarization)
