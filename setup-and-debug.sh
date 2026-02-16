# before execute it, please "pip install uv"

export UV_HTTP_TIMEOUT=300

your_k8s_config_path='/mnt/bn/trae-research-models/xujunjielong/data/config'

# IMPORTANT: if use BYTED cluster, set this to true
use_byted_venv=true

# BYTED: set proxy for connections to internet
if [ "$use_byted_venv" = true ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
    export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
fi

# uv venv setup (rllm+r2egym)
[ -d ".venv" ] || uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[verl]"
# if use byted cluster, install bytedray and byted-wandb
if [ "$use_byted_venv" = true ]; then
    export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
    uv pip uninstall ray wandb bytedray byted-wandb
    uv pip install bytedray[default,data,serve,bytedance] byted-wandb
fi

# install k8s (if not installed)
if ! command -v kubectl &>/dev/null; then
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x ./kubectl
    sudo mv ./kubectl /usr/local/bin/kubectl
fi
mkdir -p ~/.kube
# BYTERD: copy k8s config
cp $your_k8s_config_path ~/.kube/config