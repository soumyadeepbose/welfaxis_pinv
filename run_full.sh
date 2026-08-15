#!/usr/bin/env bash
# Phases B / C / D on the pod (A40 48GB), Qwen3-4B-Instruct-2507.
#
# Budget discipline: the pod is only needed for `extract` and `steer`. Run this
# with a phase argument and STOP THE POD between phases.
#
#   ./run_full.sh extract    # phase B, ~1h on A40, then stop the pod
#   ./run_full.sh gate       # phase C, local/free -- run before buying phase D
#   ./run_full.sh steer      # phase D, ~2h on A40, then stop the pod
#   ./run_full.sh report     # phase E, local -- analysis + figures
#   ./run_full.sh scale      # optional: extraction only on Qwen3-1.7B
set -euo pipefail
cd "$(dirname "$0")"

export VOID_MODEL="${VOID_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export VOID_DTYPE="${VOID_DTYPE:-bfloat16}"
export VOID_N_PAIRS="${VOID_N_PAIRS:-200}"
export VOID_N_MMLU="${VOID_N_MMLU:-500}"
export VOID_N_MMLU_TRANSFER="${VOID_N_MMLU_TRANSFER:-150}"
export VOID_B_BOOTSTRAP="${VOID_B_BOOTSTRAP:-200}"
export VOID_EXTRACT_BATCH="${VOID_EXTRACT_BATCH:-16}"
export VOID_STEER_BATCH="${VOID_STEER_BATCH:-8}"

phase="${1:-help}"
case "$phase" in
  extract)
    echo "=== phase B: extraction on $VOID_MODEL ==="
    python tests/test_contrasts.py
    python tests/test_steering_hook.py
    python extract.py --stage all
    echo "=== phase B done. STOP THE POD. Vectors + layer sweep are on disk."
    ;;
  gate)
    echo "=== phase C: null gate, geometry, variance (no GPU) ==="
    python analyze.py --skip-transfer
    python plots.py --only fig2
    python plots.py --only fig4
    python - <<'PY'
import json, pathlib, sys, os
p = pathlib.Path(os.environ.get("VOID_RESULTS", "results")) / "null_gate.json"
g = json.loads(p.read_text())
if not g["gate_passed"]:
    print("\nGATE FAILED -- do not buy phase D. Switch to the section-7 fallback\n"
          "(emotion-concept vectors, PC1 as the valence axis) and write it up as a\n"
          "designed two-method convergence check.\n")
    sys.exit(2)
print("\nGate passed. Phase D is worth the pod hours.\n")
PY
    ;;
  steer)
    echo "=== phase D: steering sweep ==="
    python steer.py
    echo "=== phase D done. STOP THE POD IMMEDIATELY."
    ;;
  report)
    echo "=== phase E: analysis + figures (no GPU) ==="
    python analyze.py
    python plots.py
    ;;
  scale)
    echo "=== optional: extraction only at a second model size ==="
    VOID_MODEL="Qwen/Qwen3-1.7B" \
    VOID_CACHE="$PWD/cache/scale17" VOID_RESULTS="$PWD/results/scale17" \
      python extract.py --stage all
    VOID_CACHE="$PWD/cache/scale17" VOID_RESULTS="$PWD/results/scale17" \
      python analyze.py --skip-transfer
    python analyze.py --scale-trend results/scale17/geometry.json results/geometry.json
    ;;
  *)
    sed -n '1,14p' "$0"
    exit 1
    ;;
esac
