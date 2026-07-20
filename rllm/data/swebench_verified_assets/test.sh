#!/bin/bash
# In-sandbox SWE-bench Verified verifier (static across instances).
#
# The ShellScriptEvaluator runs this with cwd=/testbed (the task workdir).
# Per-instance knobs live in /tests/instance.json and /tests/eval.sh; this
# script only orchestrates:
#
#   1. Capture the agent's diff vs base_commit (diff the index against
#      base_commit, not HEAD — the sweb.eval images don't guarantee HEAD is
#      at base_commit).
#   2. Hard-reset /testbed to a clean base_commit.
#   3. Re-apply the agent's diff (its candidate fix) — mirrors SWE-bench's
#      "apply model patch to clean base" grading flow.
#   4. Run /tests/eval.sh — swebench's per-instance eval script: reinstalls
#      the repo (picking up the fix), resets the test files, applies the gold
#      test_patch, and runs the repo's own test command, bracketing the output
#      with the swebench Start/End markers. It does NOT touch source files, so
#      the agent's fix survives.
#   5. grade.py parses the marked section with the vendored swebench parser and
#      rewards 1.0 iff every FAIL_TO_PASS and PASS_TO_PASS test passes.
set -uo pipefail

mkdir -p /tmp/rllm /logs/verifier
REWARD_JSON=/tmp/rllm/reward.json

log() { echo "[verifier] $*"; }

write_failure() {
    python3 - "$1" <<'PY' || echo '{"reward": 0.0, "is_correct": false}' >"$REWARD_JSON"
import json, sys
json.dump({"reward": 0.0, "is_correct": False, "metadata": {"error": sys.argv[1]}}, open("/tmp/rllm/reward.json", "w"))
PY
}

INSTANCE_JSON=/tests/instance.json
[ -f "$INSTANCE_JSON" ] || { write_failure "tests/instance.json missing"; exit 0; }
[ -f /tests/eval.sh ] || { write_failure "tests/eval.sh missing"; exit 0; }

cd /testbed 2>/dev/null || { write_failure "/testbed missing"; exit 0; }
git config --global --add safe.directory /testbed 2>/dev/null || true

BASE_COMMIT="$(python3 -c "import json; print(json.load(open('$INSTANCE_JSON')).get('base_commit', ''))")"
[ -n "$BASE_COMMIT" ] || { write_failure "base_commit empty"; exit 0; }

# Step 1: capture the agent's edits as a patch vs base_commit.
log "Capturing agent diff vs base_commit ($BASE_COMMIT)"
MODEL_PATCH=/tmp/model_patch.diff
git add -A . >/dev/null 2>&1 || true
git diff --cached --binary "$BASE_COMMIT" >"$MODEL_PATCH" 2>/dev/null || true
git reset >/dev/null 2>&1 || true
log "Captured $(wc -c <"$MODEL_PATCH" 2>/dev/null || echo 0) bytes"

# Step 2: reset to a clean base_commit so the patch applies deterministically.
git reset --hard "$BASE_COMMIT" >/dev/null 2>&1 || log "git reset --hard failed (continuing)"
git checkout "$BASE_COMMIT" >/dev/null 2>&1 || log "git checkout failed (continuing)"
git clean -fd >/dev/null 2>&1 || log "git clean failed (continuing)"

# Step 3: re-apply the agent's candidate fix.
if [ -s "$MODEL_PATCH" ]; then
    git apply -v "$MODEL_PATCH" 2>&1 | tail -20 || log "git apply failed — tests run against base"
else
    log "No agent changes detected"
fi

# Step 4: run swebench's eval script (reinstall + reset test files + test_patch
# + repo test command). Output is captured for the grader.
log "Running eval.sh"
chmod +x /tests/eval.sh 2>/dev/null || true
bash /tests/eval.sh >/tmp/eval_output.txt 2>&1 || log "eval.sh exited non-zero (grader inspects output)"

# Step 5: parse + score.
if ! python3 /tests/grade.py /tmp/eval_output.txt "$INSTANCE_JSON" "$REWARD_JSON" 2>/tmp/grade_err.txt; then
    write_failure "grade.py failed: $(tail -3 /tmp/grade_err.txt 2>/dev/null | tr '\n' ' ')"
fi
