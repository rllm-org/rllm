#!/bin/bash
# Patch verl rl_dataset.py to fix extra_info parsing
# (issue: https://github.com/rllm-org/rllm/issues/364)
#
# Usage: source scripts/patch_verl.sh

uv run --no-sync python3 -c "
import verl.utils.dataset.rl_dataset as m
f = m.__file__
with open(f, 'r') as file:
    content = file.read()
if 'import json' not in content:
    content = 'import json\n' + content
content = content.replace(
    'index = row_dict.get(\"extra_info\", {}).get(\"index\", 0)',
    'index = json.loads(row_dict.get(\"extra_info\", \"{}\")).get(\"index\", 0)'
)
content = content.replace(
    'tools_kwargs = row_dict.get(\"extra_info\", {}).get(\"tools_kwargs\", {})',
    'tools_kwargs = json.loads(row_dict.get(\"extra_info\", \"{}\")).get(\"tools_kwargs\", {})'
)
content = content.replace(
    'interaction_kwargs = row_dict.get(\"extra_info\", {}).get(\"interaction_kwargs\", {})',
    'interaction_kwargs = json.loads(row_dict.get(\"extra_info\", \"{}\")).get(\"interaction_kwargs\", {})'
)
content = content.replace(
    'need_tools_kwargs = row_dict.get(\"extra_info\", {}).get(\"need_tools_kwargs\", self.need_tools_kwargs)',
    'need_tools_kwargs = json.loads(row_dict.get(\"extra_info\", \"{}\")).get(\"need_tools_kwargs\", self.need_tools_kwargs)'
)
with open(f, 'w') as file:
    file.write(content)
print(f'Patched {f}')
"
