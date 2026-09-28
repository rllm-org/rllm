# Experiment entry points

Canonical root: `/home/yanan/agents/rllm/exp_scripts`. All personal experiment code belongs on `prpo`; do not modify `main` for experiments. This directory is orchestration, not a replacement for rLLM/veRL training code.

## Paths and storage

- Active Python/Bash entry points now use the canonical root instead of the former `gitlab/tail/rllm` and `agents/finqa-grpo-run` roots.
- `/home/yanan/agents/gitlab/tail/rllm` is a compatibility symlink to this directory, not a second code checkout. It preserves old external callers and running jobs. Historical reports retain their original paths.
- `finqa-prpo-run/checkpoints` points to `/mnt/disk1t/finqa-prpo-run-checkpoints` on the local host. The old nested destination was absent; the replacement directory is empty. This repair does **not** recover historical checkpoints. Do not resume until the selected checkpoint actually exists.
- `/mnt/disk1t` and other local checkpoint roots are machine-local. A symlink shared through NFS does not transfer checkpoint contents to another host.
- New training output should remain on the training host's local disk. Models, datasets, checkpoints, merged weights, logs, trajectories, environments and credentials are excluded from Git. Scripts and small configuration files remain eligible.

## Environment status (2026-09-26)

The default remains `/home/yanan/agents/rllm/.venv`. No packages were changed in this environment during the path repair. `finqa-grpo-run/env.sh` also configures the CUDA shared-library loader and judge settings; source it when using the associated scripts. Its private `.azure_creds` file must remain untracked and must not be printed.

Installed versions include Python 3.11.13, rLLM 0.3.0rc0, veRL 0.8.0, vLLM 0.22.1, PyTorch 2.11.0+cu129, Transformers 5.5.4, PEFT 0.20.0 and FlashAttention 2.8.3. `environment/requirements-legacy.txt` is an installed-package snapshot, **not** a successfully resolved installation lockfile. Host CUDA drivers, model caches and private credentials are not captured by that snapshot.

CPU/import checks pass, including `vllm._C` and `flash_attn`, but dependency declarations conflict:

| Package | Relevant requirement / installed value |
| --- | --- |
| veRL 0.8.0 | NumPy <2 |
| vLLM 0.22.1 | opencv-python-headless >=4.13.0 |
| Installed OpenCV 5.0.0.93 | NumPy >=2 on Python 3.11 |
| Installed ml-dtypes 0.6.0 | NumPy >=2 |
| mistral-common 1.11.7 | NumPy <2.4 on Python 3.11; installed NumPy is 2.4.6 |

An isolated NumPy-1.26/OpenCV-4.11/ml-dtypes-0.5 probe passed CPU imports but violated vLLM's OpenCV requirement. It was rejected, not promoted. Its package snapshot is `environment/requirements-rejected-numpy1-probe.txt`; do not use it as a recommended environment. Fixing the whole constraint set requires choosing different framework versions, not merely changing NumPy. That change needs a separately validated protocol and is not part of the path repair. Existing successful execution is not evidence that package declarations are consistent.

## Preflight

```bash
cd /home/yanan/agents/rllm
source exp_scripts/finqa-grpo-run/env.sh
"$VENV/bin/python" exp_scripts/environment/check_environment.py
```

This checks CPU-only imports, script syntax, canonical paths and package declarations without starting training or opening a CUDA context. By default known dependency conflicts make it fail. For a diagnostic of the retained historical stack, `--allow-dependency-conflicts` reports them without blocking; this is an explicit waiver, not a dependency repair.

Before any formal run, additionally verify the intended task split/count, cached model revision, sampling parameters, algorithm namespace overrides, local checkpoint destination, free disk/GPU memory and host-specific grader concurrency. Syntax/import checks do not replace a hardware smoke test. Historical scripts may launch or delete artifacts; do not run an entry point merely because it exists.
