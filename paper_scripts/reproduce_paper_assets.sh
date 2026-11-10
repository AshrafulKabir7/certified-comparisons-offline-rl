#!/usr/bin/env bash
# Regenerate every table (.tex), data figure and facts file of the paper from the stored results of the final run,
# then compare them with the files used in the paper (paper_assets_expected/). No experiment is rerun.
# Usage: bash paper_scripts/reproduce_paper_assets.sh [OUT_DIR]      (default: ./regenerated)
#        PYTHON=/path/to/python bash paper_scripts/reproduce_paper_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/regenerated}" && cd "${1:-$ROOT/regenerated}" && pwd)"
PY="${PYTHON:-python3}"
T="$ROOT/results/tables"
mkdir -p "$OUT/scripts" "$OUT/tables" "$OUT/figures"
cp "$ROOT"/paper_scripts/*.py "$OUT/scripts/"   # verify_bracket.py writes next to its own folder: $OUT
rm -rf "$OUT/original_run" && cp -R "$ROOT/paper_assets_expected/original_run" "$OUT/original_run"

echo "[1/9] tables extracted from the final-run record"
"$PY" -I "$OUT/scripts/extract_v3.py" "$ROOT/run_record/Sepsis_Offline_RL_v4_final_run.ipynb" "$OUT/v3_tables"
echo "[2/9] full-precision files of the additional experiments"
"$PY" -I "$OUT/scripts/use_exact_v3.py" "$OUT" "$T/xenv"
echo "[3/9] tables of the additional experiments and facts_v3.json"
"$PY" -I "$OUT/scripts/make_v3_assets.py" "$OUT"
echo "[4/9] Parts I and II: tables, figures, facts.json"
"$PY" -I "$OUT/scripts/make_assets.py" "$T" "$OUT"
echo "[5/9] robustness analyses and facts_robust.json"
"$PY" -I "$OUT/scripts/make_robustness.py" "$T" "$OUT"
echo "[6/9] confidence sets, sensitivity cells and facts_v4.json"
"$PY" -I "$OUT/scripts/make_v4_assets.py" "$T" "$OUT"
echo "[7/9] final run vs earlier run and facts_repro.json"
"$PY" -I "$OUT/scripts/make_repro_summary.py" "$OUT" > /dev/null
echo "[8/9] numerical check of the bracketing theorem on random MDPs (about 100 s)"
"$PY" -I "$OUT/scripts/verify_bracket.py"
echo "[9/9] comparison with paper_assets_expected/"
"$PY" -I "$ROOT/paper_scripts/compare_assets.py" "$OUT" "$ROOT/paper_assets_expected" "$ROOT/results/v3_tables"
