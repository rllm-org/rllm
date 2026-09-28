#!/usr/bin/env bash
# FinQA chain status. Two things are alerted on, both of which stay invisible in
# TerminationReason because the flow swallows them:
#   * turn exhaustion  -> metrics.steps_used == MAX_TURNS
#   * context overflow -> MAX_PROMPT_LENGTH_EXCEEDED / context-length errors
RD=/home/yanan/agents/rllm/exp_scripts/finqa-grpo-run/eval
echo "--- chain log ---"; tail -6 "$RD/chain-20260922.log" 2>/dev/null
ps -eo cmd | grep -q "[f]inqa_chain.sh" && echo "chain=alive" || echo "chain=GONE"
for tag in 2507-fair-t20 qwen35-nothink-t50 qwen35-think-t50; do
  D="$RD/$tag"; [ -d "$D" ] || continue
  echo "--- $tag ---"
  for s in val test; do
    if [ -f "$D/$s.json" ]; then
      python3 - "$D/$s.json" "$s" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("  %-5s DONE %.2f%% (%d/%d) errors=%d" % (sys.argv[2],100*d["correct"]/d["total"],d["correct"],d["total"],d["errors"]))
PY
    else
      echo "  $s   $(grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' "$D/eval_$s.log" 2>/dev/null | tail -1)"
    fi
    # context overflow: fatal for run C, so counted per split and alerted
    L="$D/eval_$s.log"
    if [ -f "$L" ]; then
      ctx=$(grep -icE "MAX_PROMPT_LENGTH_EXCEEDED|maximum context length|context_length_exceeded|reduce the length" "$L")
      tot=$(grep -c "Rollout completed" "$L")
      if [ "${tot:-0}" -gt 0 ] && [ "${ctx:-0}" -gt 0 ]; then
        pct=$(( 100 * ctx / tot ))
        echo "    CONTEXT_OVERFLOW: $ctx/$tot = ${pct}%"
        [ "$pct" -gt 5 ] && echo "    ALERT_CONTEXT $tag/$s ${pct}%"
      fi
    fi
  done
  # The cap lives in the tag (t20 / t50); "at max observed" is NOT the same as
  # "hit the cap", and conflating them hides or invents turn exhaustion.
  CAP=$(echo "$tag" | grep -oE "t[0-9]+$" | tr -d t); CAP=${CAP:-20}
  python3 - "$D" "$CAP" <<'PY'
import json,glob,os,sys,collections
d=sys.argv[1]; cap=int(sys.argv[2]); c=collections.Counter(); n=0
for f in glob.glob(os.path.join(d,"**","*.json"),recursive=True):
    if os.path.basename(f) in ("val.json","test.json","protocol.json"): continue
    try: e=json.load(open(f))
    except Exception: continue
    s=(e.get("metrics") or {}).get("steps_used")
    if s is None: continue
    c[s]+=1; n+=1
if n:
    mx=max(c); at_cap=sum(v for k,v in c.items() if k>=cap)
    print("  turns: n=%d cap=%d max_used=%d at_cap=%d (%.1f%%)" % (n,cap,mx,at_cap,100*at_cap/n))
    if 100*at_cap/n > 5: print("  ALERT_TURN_EXHAUSTION %.1f%%" % (100*at_cap/n))
PY
done
