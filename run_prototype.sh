#!/usr/bin/env bash
# Phase A -- free. Colab T4, Qwen3-0.6B, tiny N, full pipeline end to end.
#
# Purpose is to catch chat-template bugs, hook-placement errors and the `bare`
# tokenisation path. Do not proceed to Phase B until this produces all four
# figures, however ugly they are.
set -euo pipefail
cd "$(dirname "$0")"

export VOID_MODEL="${VOID_MODEL:-Qwen/Qwen3-0.6B}"
export VOID_DTYPE="${VOID_DTYPE:-float16}"     # T4 has no bf16
export VOID_N_PAIRS=20
export VOID_N_MMLU=50
export VOID_N_MMLU_TRANSFER=50
export VOID_B_BOOTSTRAP=20
export VOID_B_STEER=3
export VOID_N_MMLU_BOOT=20
export VOID_N_COHERENCE=8
export VOID_EXTRACT_BATCH="${VOID_EXTRACT_BATCH:-8}"
export VOID_STEER_BATCH="${VOID_STEER_BATCH:-4}"
export VOID_CACHE="${VOID_CACHE:-$PWD/cache/proto}"
export VOID_RESULTS="${VOID_RESULTS:-$PWD/results/proto}"

echo "=== phase A: prototype on $VOID_MODEL ==="
python tests/test_contrasts.py
python tests/test_steering_hook.py
python tests/test_pipeline_synthetic.py

echo "--- extraction"
python extract.py --stage all

echo "--- null-contrast gate + geometry (no GPU needed)"
python analyze.py --skip-transfer

echo "--- steering sweep"
python steer.py

echo "--- analysis + figures"
python analyze.py
python plots.py

echo "=== phase A done. Inspect:"
ls -la "$VOID_RESULTS" | sed 's/^/    /'
echo "Check in this order: tokenisation_report.json (bare path distinct?),"
echo "contrast_audit.json (token deltas), layer_sweep.json (is L* interior?),"
echo "null_gate.json, then the four figures."
