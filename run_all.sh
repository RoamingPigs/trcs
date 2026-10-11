#!/usr/bin/env bash
# The full run, in the order used for the published results. Resumable.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
DATA="${DATA:-data/trcs}"
OUT="${OUT:-results}"
python3 run_e2.py --data "$DATA" --out "$OUT/e2"
python3 run_e1.py --data "$DATA" --out "$OUT/e1"
python3 run_speech_cost.py --data "$DATA" --out "$OUT/speech_cost"
echo "ALL RUNS DONE"
