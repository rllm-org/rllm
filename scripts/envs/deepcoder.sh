# env plugin: deepcoder — code generation + firejail sandbox eval
COOKBOOK_DIR="cookbooks/deepcoder"
DATASETS=""
EXTRA_DEPS="datasets"
EXTRA_APT="firejail"
PREPARE_CMD="python cookbooks/deepcoder/prepare_deepcoder_data.py"
