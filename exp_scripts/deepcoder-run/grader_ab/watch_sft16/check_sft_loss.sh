#!/usr/bin/env bash
# One read-only snapshot of the .16 self-SFT run, focused on the loss curve.
H=10.225.68.16
L=/home/yanan/.deepcoder-sft-pipeline-formal/sft-current4360-e4-len18000.log
D=/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-current4360-r32-e4-len18000
SPE=136   # steps per epoch (4360 samples / batch 32, drop_last)
TOT=544   # 4 epochs

ssh -o ConnectTimeout=10 "$H" "
L=$L; D=$D; SPE=$SPE; TOT=$TOT
echo \"T=\$(date '+%F %T %Z')\"

if pgrep -f verl_entry >/dev/null 2>&1; then ALIVE=yes; else ALIVE=no; fi
STEP=\$(grep -o 'step:[0-9]*' \"\$L\" | tail -1 | cut -d: -f2)
STEP=\${STEP:-0}
EP=\$(( (STEP + SPE - 1) / SPE )); [ \$EP -lt 1 ] && EP=1
echo \"alive=\$ALIVE step=\${STEP}/\$TOT epoch=\${EP}/4\"

# per-epoch loss statistics -- the signal that matters
echo 'EPOCH_LOSS (mean/min/max/n):'
grep -o 'step:[0-9]* - .*train/loss:[0-9.e+-]*' \"\$L\" \
 | sed 's/step:\([0-9]*\).*train\/loss:\([0-9.e+-]*\)/\1 \2/' \
 | awk -v spe=\$SPE '{e=int((\$1-1)/spe)+1; s[e]+=\$2; n[e]++; if(mn[e]==\"\"||\$2<mn[e])mn[e]=\$2; if(\$2>mx[e])mx[e]=\$2}
     END{for(i=1;i<=4;i++) if(n[i]>0) printf \"  epoch %d: mean=%.4f min=%.4f max=%.4f n=%d\n\", i, s[i]/n[i], mn[i], mx[i], n[i]}'

# trend inside the most recent window
echo -n 'LAST_10_LOSS: '
grep -o 'train/loss:[0-9.e+-]*' \"\$L\" | tail -10 | sed 's/.*://' | awk '{printf \"%.3f \", \$1} END{print \"\"}'
echo -n 'GRAD_NORM(min/max all): '
grep -o 'train/grad_norm:[0-9.e+-]*' \"\$L\" | sed 's/.*://' | sort -n | awk 'NR==1{a=\$1} END{printf \"%.3f / %.3f\n\", a, \$1}'

# health
NAN=\$(grep -c -i 'loss:nan\|loss:inf' \"\$L\")
TB=\$(grep -c '^Traceback' \"\$L\")
echo \"NAN_OR_INF=\$NAN TRACEBACKS=\$TB\"
echo -n 'PACE: '; tail -1 \"\$L\" | grep -o '[0-9.]*s/it' | tail -1
echo -n 'CKPTS: '; ls -1 \"\$D\" 2>/dev/null | grep -c global_step; ls -1 \"\$D\" 2>/dev/null | grep global_step | tr '\n' ' '; echo
echo -n 'GPU: '; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
if [ \"\$ALIVE\" = no ]; then
  if [ \"\$STEP\" -ge \"\$TOT\" ]; then echo 'STATUS=COMPLETE'; else echo 'FATAL: verl_entry gone at step '\$STEP' of '\$TOT; fi
fi
"
