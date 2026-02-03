export UV_HTTP_TIMEOUT=300
export UV_INDEX_URL=https://bytedpypi.byted.org/simple/ # if using byted cluster
export HF_ENDPOINT=https://hf-mirror.com
export KUBE_NAMESPACE="rllm"

your_k8s_config_path='/mnt/bn/trae-research-models-lq/xujunjielong/data/config' # if using byted cluster. please use your own config
env_debug_script=debug_swe_minimal.py
use_byted_venv=true # set to false if you want to use system python env

# uv venv setup (rllm+r2egym)
[ -d ".venv" ] || uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[verl]"
cd R2E-Gym
uv pip install -e .
cd ..
# if use byted cluster, install bytedray and byted-wandb
[ "$use_byted_venv" = true ] && uv pip install bytedray[default,data,serve,bytedance] byted-wandb

# install k8s (if not installed)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
mkdir -p ~/.kube
cp $your_k8s_config_path ~/.kube/config # if using byted cluster. please use your own config
kubectl create namespace $KUBE_NAMESPACE || echo "namespace $KUBE_NAMESPACE already exists, skipping"
kubectl config set-context --current --namespace=$KUBE_NAMESPACE

# the script to be run
uv run python $env_debug_script