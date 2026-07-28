# GLM-5.2 OpenCode medium-task curriculum

## Dataset

- 48 fixed `tb-opus-pass` tasks.
- Each task scored 3–5 successes in eight OpenCode rollouts.
- All eight rollouts completed with `ENV_DONE`; exclude timeouts and infrastructure, model, sandbox, setup, verifier, and grading failures.
- Training and Terminal-Bench 2.1 task IDs must not overlap.

## Training

- Full-parameter `accounts/fireworks/trainingShapes/glm-5p2-200k`.
- Strict synchronous on-policy GRPO.
- 16 distinct prompts/groups per optimizer step.
- Eight rollouts per group: 128 trajectories per step.
- Three optimizer steps per epoch; four epochs; 12 optimizer steps total.
- Two trainer replicas plus six rollout replicas: 10 nodes.
- Keep the production optimizer, sampling, filtering, context, and timeout settings.

## Evaluation

- Do not evaluate before training.
- Evaluate all 89 `terminal-bench@2.1` tasks only after steps 3, 6, 9, and 12.
- Do not duplicate the final step-12 evaluation.

## Launch

```bash
bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh \
  full opencode curriculum
```
