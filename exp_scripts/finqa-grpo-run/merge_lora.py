"""Turn a verl LoRA checkpoint into a servable, genuinely-merged HF model.

Why this exists
---------------
`python -m verl.model_merger merge --backend fsdp` does NOT produce a merged
model when LoRA is in play. In `base_model_merger.save_hf_model_and_tokenizer`:

    lora_path = self.save_lora_adapter(state_dict)   # strips LoRA keys IN PLACE
    ...
    model.save_pretrained(target_dir, state_dict=state_dict)   # saves what's LEFT

so `target_dir/` ends up holding the **base** weights and the adapter is written
separately to `target_dir/lora_adapter/`. Serving `target_dir` with vLLM would
evaluate the base model and report it as the trained result — a plausible-looking
number that means nothing.

This script runs the merger, then applies the adapter with PEFT
(`merge_and_unload`), and refuses to emit anything unless the resulting weights
actually differ from the base.

Usage
-----
    python merge_lora.py <checkpoint_global_step_dir> <output_dir>

e.g. python merge_lora.py \
        ~/agents/rllm/exp_scripts/finqa-grpo-run/checkpoints/qwen3-4b-16-epoch/global_step_125 \
        ~/agents/rllm/exp_scripts/finqa-grpo-run/merged/step_125
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def remove_staging(path: Path, *, required: bool) -> None:
    """Remove an NFS staging tree, tolerating delayed directory visibility."""
    for attempt in range(20):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 19:
                if required:
                    raise
                print(f"[merge] WARNING: merged model is complete but stale staging "
                      f"could not be removed: {path}", file=sys.stderr)
                return
            time.sleep(0.25)


def die(what: str, expected: str, actual: str, why: str, fix: str) -> None:
    print(f"\nMERGE FAILED: {what}", file=sys.stderr)
    print(f"  expected: {expected}", file=sys.stderr)
    print(f"  actual  : {actual}", file=sys.stderr)
    print(f"  why     : {why}", file=sys.stderr)
    print(f"  fix     : {fix}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    ckpt = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve()
    actor = ckpt / "actor"

    if not actor.is_dir():
        die("checkpoint layout", f"{actor} exists", "missing",
            "verl writes shards under <global_step_N>/actor/",
            "pass the global_step_N directory, not the run directory")

    staging = out.parent / (out.name + "_verl_raw")
    if staging.exists():
        remove_staging(staging, required=True)
    if out.exists():
        shutil.rmtree(out)
    staging.parent.mkdir(parents=True, exist_ok=True)

    # ---- step 1: verl merger (base weights -> staging/, adapter -> staging/lora_adapter/)
    print(f"[merge] running verl model_merger on {actor}")
    subprocess.run(
        [sys.executable, "-m", "verl.model_merger", "merge",
         "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(staging)],
        check=True,
    )

    adapter = staging / "lora_adapter"
    if not adapter.is_dir():
        die("no LoRA adapter produced", "staging/lora_adapter/ exists", "missing",
            "either this checkpoint is not a LoRA run, or lora_train_meta.json was absent",
            "check <ckpt>/actor/lora_train_meta.json; if training was full-parameter, "
            "serve the merger output directly and skip this script")

    # ---- step 2: actually apply the adapter
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] loading base from {staging}")
    base = AutoModelForCausalLM.from_pretrained(staging, torch_dtype=torch.bfloat16)

    # keep a reference copy of one adapted layer so the merge can be proven
    from peft import PeftConfig
    target_names = set(PeftConfig.from_pretrained(adapter).target_modules or [])
    probe_name, probe_before = None, None
    for name, param in base.named_parameters():
        if any(t in name for t in target_names) and param.dim() == 2:
            probe_name, probe_before = name, param.detach().clone()
            break
    if probe_name is None:
        # 'all-linear' style config: fall back to any 2-D weight in the decoder
        for name, param in base.named_parameters():
            if "layers.0." in name and param.dim() == 2:
                probe_name, probe_before = name, param.detach().clone()
                break

    print(f"[merge] applying adapter from {adapter}")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()

    # ---- step 3: prove the weights moved
    probe_after = dict(merged.named_parameters()).get(probe_name)
    if probe_after is None:
        die("probe weight vanished after merge", f"{probe_name} present", "missing",
            "merge_and_unload renamed or dropped the tensor", "inspect the PEFT version")
    if torch.equal(probe_before, probe_after.detach().cpu()):
        die("merged weights are identical to base",
            f"{probe_name} changed", "bitwise identical",
            "the adapter did not apply — serving this would silently evaluate the BASE model, "
            "which is exactly the failure this script exists to prevent",
            "check that <ckpt>/actor/lora_train_meta.json has r>0 and that the adapter "
            "safetensors are non-empty")
    delta = (probe_after.detach().cpu().float() - probe_before.float()).abs()
    print(f"[merge] VERIFIED: {probe_name} changed  "
          f"(mean |delta| = {delta.mean():.3e}, max = {delta.max():.3e})")

    # ---- step 4: save, with the tokenizer the checkpoint shipped
    print(f"[merge] saving merged model to {out}")
    merged.save_pretrained(out)
    hf_src = actor / "huggingface"
    tok_src = hf_src if (hf_src / "tokenizer.json").exists() else staging
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out)

    # NFS may report ENOTEMPTY briefly after the last child was unlinked.  The
    # merged output is already verified and complete, so cleanup must not turn
    # a successful merge into a failed evaluation queue item.
    remove_staging(staging, required=False)
    print(f"[merge] DONE -> {out}")


if __name__ == "__main__":
    main()
