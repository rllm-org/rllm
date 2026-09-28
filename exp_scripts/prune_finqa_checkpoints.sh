#!/usr/bin/env bash
# User-approved FinQA retention. Run only after the verified migration completes.
set -euo pipefail
ROOT=/home/yanan/agents/rllm/exp_scripts
MIGRATION_PID=1180264
while kill -0 "$MIGRATION_PID" 2>/dev/null; do sleep 10; done
for PATH_TO_CHECK in "$ROOT/finqa-grpo-run/merged" "$ROOT/finqa-prpo-run/checkpoints" "$ROOT/finqa-reinforce_plus_plus_baseline-run/checkpoints"; do
  test -L "$PATH_TO_CHECK" && test -d "$PATH_TO_CHECK"
  [[ "$(readlink -f "$PATH_TO_CHECK")" == /mnt/disk1t/* ]]
done
delete_target() {
  local TARGET="$1" RESOLVED
  test -d "$TARGET" && test ! -L "$TARGET"
  RESOLVED=$(realpath "$TARGET")
  case "$RESOLVED" in
    /mnt/disk1t/finqa-grpo-run-checkpoints/*/global_step_*|/mnt/disk1t/finqa-grpo-run-merged/global_step_*|/mnt/disk1t/finqa-prpo-run-checkpoints/raw/global_step_*|/mnt/disk1t/finqa-prpo-run-checkpoints/merged-eval-models/global_step_*|/mnt/disk1t/finqa-prpo-run-checkpoints/checkpoints/*/global_step_*|/mnt/disk1t/finqa-reinforce_plus_plus_baseline-run-checkpoints/*/global_step_*) ;;
    *) echo "Refusing unexpected target $RESOLVED"; exit 1 ;;
  esac
  echo "DELETE $RESOLVED"
  du -sh "$RESOLVED"
  rm -rf -- "$RESOLVED"
}
while IFS= read -r TARGET; do
  NAME=$(basename "$TARGET")
  [[ "$NAME" =~ ^global_step_([0-9]+)(_verl_raw)?$ ]] || exit 1
  case "${BASH_REMATCH[1]}" in 620|651|682|713) echo "KEEP $TARGET" ;; *) delete_target "$TARGET" ;; esac
done < <(find /mnt/disk1t/finqa-grpo-run-checkpoints /mnt/disk1t/finqa-grpo-run-merged -maxdepth 2 -type d -name 'global_step_*')
while IFS= read -r TARGET; do
  NAME=$(basename "$TARGET")
  [[ "$NAME" =~ ^global_step_([0-9]+)$ ]] || exit 1
  case "${BASH_REMATCH[1]}" in 992|1054|1116|1178|1240|1302) echo "KEEP $TARGET" ;; *) delete_target "$TARGET" ;; esac
done < <(find /mnt/disk1t/finqa-prpo-run-checkpoints/raw /mnt/disk1t/finqa-prpo-run-checkpoints/merged-eval-models /mnt/disk1t/finqa-prpo-run-checkpoints/checkpoints -maxdepth 2 -type d -name 'global_step_*')
while IFS= read -r TARGET; do delete_target "$TARGET"; done < <(find /mnt/disk1t/finqa-reinforce_plus_plus_baseline-run-checkpoints -maxdepth 2 -type d -name 'global_step_*')
echo 'COMPLETE: FinQA pruning; result/log artifacts untouched.'
df -h / /mnt/disk1t
