# env plugin: codex_code — Codex CLI agent in bubblewrap sandbox
COOKBOOK_DIR=""
DATASETS=""
EXTRA_DEPS="tensordict>=0.10 transformers>=5.0"
EXTRA_APT="bubblewrap"
PREPARE_CMD=""

# Codex CLI inside bwrap needs network to reach gateway
export RLLM_BWRAP_NETWORK=1

env_setup() {
    _default_env_setup

    # Pre-install Node.js 22 + Codex CLI on host
    # (visible inside bwrap via /usr read-only bind-mount)
    if ! command -v codex &>/dev/null; then
        echo "  Installing Node.js 22 + Codex CLI..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
        apt-get install -y -qq nodejs
        npm install -g @openai/codex
    fi
    echo "  codex: $(codex --version 2>/dev/null || echo 'not found')"
}
