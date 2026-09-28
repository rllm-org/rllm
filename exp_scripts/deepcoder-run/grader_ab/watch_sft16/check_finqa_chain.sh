#!/usr/bin/env bash
# Status of the three-run FinQA chain on .24, with turn-exhaustion accounting.
# steps_used == MAX_TURNS is the real exhaustion signal; TerminationReason stays
# ENV_DONE because the flow falls back to the last assistant message.
echo "T=$(date '+%F %T %Z')"
ssh -o ConnectTimeout=10 10.225.68.24 'bash /tmp/finqa_chain_check.sh'
