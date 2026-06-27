#!/usr/bin/env bash
#
# Reproduce the Stage B sandwiched backup-CBF results and export them to
# examples/rear_aware/saved_results/sandwich_bcbf/.
#
# Demonstrates that interaction-model accuracy decides SAFETY for the
# rear-aware backup CBF (SandwichedBackupCBF1D): the ego is sandwiched between a
# mildly braking lead and a sluggish interactive rear; its backup is a follow-lead
# OVM plus a beta_f*(v_rear - v_ego) rear-aware softening term.
#
#   passive / under / accurate  -> safe on both fronts (min h_r > 0)
#   over-estimated rear         -> the ego brakes harder than the real rear can
#                                  follow -> rear-end (min h_r < 0), still forward-safe.
#
# Operating point (validated, monotone crossing near scale ~1.75):
#   actual rear OVM alpha/beta = 0.32/0.224, kappa = 0.7
#   beta_f = 1.0, u_acc = 3, gap_r0 = 4.5, mild lead brake = 1.5
#
# Run from anywhere; paths are resolved from the script's own location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

OUT="examples/rear_aware/saved_results/sandwich_bcbf"
RUN="examples/rear_aware/run_sandwich.py"
PLOT="examples/rear_aware/plot_sandwich_cases.py"

# Fresh export so the registry numbers the runs deterministically run_001..004.
rm -rf "$OUT"
mkdir -p "$OUT"

# Shared scenario: aggressive ego nominal + mild lead brake + sluggish actual rear.
COMMON=(--method sandwiched_bcbf
        --rear-alpha 0.32 --rear-beta 0.224 --rear-kappa 0.7
        --gap-r0 4.5 --backup-beta-f 1.0
        --output-dir "$OUT")

# run_001..004, in this order, sweeping how responsive the ego ASSUMES the rear is.
uv run python "$RUN" "${COMMON[@]}" --assumed-rear-alpha 0.00 --assumed-rear-beta 0.000 --note passive
uv run python "$RUN" "${COMMON[@]}" --assumed-rear-alpha 0.16 --assumed-rear-beta 0.112 --note under
uv run python "$RUN" "${COMMON[@]}"                                                       --note accurate
uv run python "$RUN" "${COMMON[@]}" --assumed-rear-alpha 0.80 --assumed-rear-beta 0.560 --note over

# Comparison spec (paths relative to the example dir, matching plot_sandwich_cases.py I/O).
cat > "$OUT/sandwich_bcbf_cases.json" <<'JSON'
{
  "output": "saved_results/sandwich_bcbf/compare_sandwich.png",
  "runs": {
    "passive (x0.0)":  "saved_results/sandwich_bcbf/run_001",
    "under (x0.5)":    "saved_results/sandwich_bcbf/run_002",
    "accurate (x1.0)": "saved_results/sandwich_bcbf/run_003",
    "over (x2.5)":     "saved_results/sandwich_bcbf/run_004"
  }
}
JSON

# 6 stacked time series + 2 phase portraits -> saved_results/sandwich_bcbf/compare_sandwich.png
uv run python "$PLOT" "$OUT/sandwich_bcbf_cases.json"

echo
echo "Exported Stage B results to $OUT/"
