# FinQA Multi-Table Evaluation Protocol v2

This directory defines a versioned AgentFlow for the 991-row `multi_train`,
126-row `multi_val`, and 131-row `multi_test` splits. It does not modify the
upstream rLLM FinQA flow, evaluator, or trainer, and its evaluation results must
not be compared numerically with the old 20-turn multi-table protocol.

Training support has been developed but is intentionally **unvalidated**:

- `train_multitable.py` selects `multi_train`/`multi_val`, checks their exact
  sizes and structure, and runs a sentinel-based private-supervision leak gate
  before constructing the public rLLM `AgentTrainer`.
- `train_16_multitable_grpo.sh` is the two-GPU veRL/GRPO launcher. It shares
  this directory's rollout flow with evaluation and saves once per whole epoch.
- The 49,152-token training envelope and provisional vLLM memory fraction have
  not received a smoke test. Do not start a formal run until those checks and
  an actual rollout transcript audit pass.

Protocol identity:

- agent protocol: `finqa-multitable-visible-complete-v2`
- maximum model turns: 50
- tool-use phase: at most 45 model turns
- reserved final-synthesis attempts: 5
- discovery response cap: 2,048 tokens per model call
- final response cap: 8,192 tokens
- serving context: 49,152 tokens
- tool output cap: 8,000 characters per call
- decoding: greedy (`temperature=0`, `top_p=1.0`, `seed=1234`)
- judge: existing upstream FinQA multi-table rubric, using `gpt-5.4-nano`

The flow is loaded through rLLM's public `module:object` extension point:

```text
--agent multitable_v2_flow:finqa_multitable_v2 --evaluator finqa
```

Every episode records the protocol version, prompt hash, turn/tool counts,
termination reason, final-answer fallback status, and serving errors.  Every
result directory also receives a manifest containing hashes of the flow,
policy prompt, and unchanged upstream judge rubric.
