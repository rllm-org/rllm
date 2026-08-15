#!/bin/bash
# Provision a CPU-only coordinator box for terminal-RL.
#
# Nothing here needs a GPU: weights, optimizer state and checkpoints live on
# Fireworks, and rollouts run in Modal sandboxes. This host runs the model
# gateway and the workflow engine.
#
#   4-8 vCPU, 16 GB RAM, 100 GB+ disk (episode logging writes ~12 GB/h
#   unbounded, ~5 GB steady-state once pruning kicks in).
#
# Usage:  bash setup.sh [--repos-only]
#
# Verifies rather than assumes: it fails loudly if an editable install did not
# land, because a stale wheel silently shadows the local checkout and you get
# confusing "why is my code change not taking effect" behaviour.

set -euo pipefail

RLLM_BRANCH="${RLLM_BRANCH:-tianyi/terminal-rl}"
ROOT="${TERMINAL_RL_ROOT:-$HOME}"
ENV_NAME="${TERMINAL_RL_CONDA_ENV:-verl}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "1/6  python env"
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install miniconda first, or create a python3.12 venv"
    echo "and re-run with the env already active."
else
    conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.12
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi
python3 -c 'import sys; assert sys.version_info[:2]>=(3,12), f"need py3.12+, got {sys.version}"'

say "2/6  repos"
# rllm carries the terminal-RL cookbook, the gateway, and the Fireworks backend.
[ -d "$ROOT/rllm" ] || git clone <RLLM_REMOTE> "$ROOT/rllm"
git -C "$ROOT/rllm" fetch --all --quiet || true
git -C "$ROOT/rllm" checkout "$RLLM_BRANCH"

# The Fireworks SDK and training cookbook are consumed as EDITABLE local
# checkouts, not from PyPI. Without them `import training.provision` and the
# tinker LoRA-alpha patch are missing.
[ -d "$ROOT/fireworks" ] || {
    echo "MISSING: $ROOT/fireworks  (fireworks monorepo checkout)"
    echo "  needs: public-repos/python-sdk, public-repos/cookbook/training,"
    echo "         cookbook/fireworks-training-infra"
    exit 1
}

say "3/6  editable installs"
pip install -e "$ROOT/rllm"
pip install -e "$ROOT/rllm/rllm-model-gateway"
pip install -e "$ROOT/fireworks/fireworks/public-repos/python-sdk"
pip install -e "$ROOT/fireworks/fireworks/public-repos/cookbook/training"
pip install -e "$ROOT/fireworks/cookbook/fireworks-training-infra"

say "4/6  verify imports resolve to the local checkouts"
python3 - <<'PY'
import importlib, sys
expect = {
    "rllm": "/rllm/",
    "rllm_model_gateway": "/rllm-model-gateway/",
    "fireworks": "/python-sdk/",
    "training": "/cookbook/training/",
}
bad = []
for mod, want in expect.items():
    try:
        f = importlib.import_module(mod).__file__ or ""
    except Exception as e:
        bad.append(f"{mod}: import failed ({type(e).__name__}: {e})"); continue
    if want not in f:
        bad.append(f"{mod}: resolved to {f} (expected a checkout containing {want})")
    else:
        print(f"  ok  {mod:20s} {f}")
if bad:
    print("\nFAILED - a wheel is shadowing a local checkout:")
    [print("  " + b) for b in bad]
    sys.exit(1)
PY

if [ "${1:-}" = "--repos-only" ]; then say "done (repos only)"; exit 0; fi

say "5/6  credentials + dataset"
mkdir -p "$HOME/.rllm/terminal-rl-logs"
if [ ! -f "$HOME/.rllm/terminal-rl-auto.env" ]; then
    cp "$ROOT/rllm/cookbooks/terminal-rl/deploy/terminal-rl.env.example" "$HOME/.rllm/terminal-rl-auto.env"
    chmod 600 "$HOME/.rllm/terminal-rl-auto.env"
    echo "  created ~/.rllm/terminal-rl-auto.env  -- EDIT IT (key, account, public URL)"
fi
echo "  modal:  copy ~/.modal.toml from the old host (keep the intended profile active)"
echo "  wandb:  copy ~/.netrc, or run 'wandb login'"
echo "  data :  copy ~/.rllm/datasets/tb_v2_debug (56K), or run prepare_data.py"

say "6/6  service"
cat <<EOF
  systemd (Linux):
    mkdir -p ~/.config/systemd/user
    cp $ROOT/rllm/cookbooks/terminal-rl/deploy/terminal-rl-sweep.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now terminal-rl-sweep
    loginctl enable-linger \$USER

  Before starting, confirm the OLD host is stopped -- two coordinators means
  two billing trainer jobs and two gateways fighting for the same rollouts.
EOF
