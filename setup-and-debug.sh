# before execute it, please "pip install uv"

export UV_HTTP_TIMEOUT=300
export KUBE_NAMESPACE="rllm"

your_k8s_config_path='/mnt/bn/trae-research-models/xujunjielong/data/config'
env_debug_script=debug_swe_minimal.py

# IMPORTANT: if use BYTED cluster, set this to true
use_byted_venv=true

# BYTED: set proxy for connections to internet
if [ "$use_byted_venv" = true ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
    export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export no_proxy="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8"
fi

# uv venv setup (rllm+r2egym)
[ -d ".venv" ] || uv venv --python 3.11
source .venv/bin/activate
echo "=== installing rLLM w./ VeRL ==="
uv pip install -e ".[verl]"
cd R2E-Gym
echo "=== installing R2E-Gym ==="
uv pip install -e .
cd ..
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
if ! kubectl get namespace $KUBE_NAMESPACE &>/dev/null; then
    kubectl create namespace $KUBE_NAMESPACE
fi
kubectl config set-context --current --namespace=$KUBE_NAMESPACE

# the script to be run
echo "=== start debugging env with uv ==="
uv run python $env_debug_script