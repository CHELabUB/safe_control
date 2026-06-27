#!/usr/bin/env bash
#
# Reproduce the Stage A sandwiched HOCBF results and export them to
# examples/rear_aware/saved_results/sandwich_hocbf/.
#
# Analytic combined forward-CBF ceiling + coupled rear-HOCBF floor with the a_e_eff
# coupling (sandwiched_cbf.SandwichedHOCBF). The ego is sandwiched between a mildly
# braking lead and an interactive rear; the assumed rear model's survivable braking
# authority (a_e_eff) is fed into the forward safe-distance, so accuracy decides safety:
#
#   passive / under / accurate  -> safe on both fronts (min h_r > 0)
#   over-estimated rear         -> the ego tailgates and brakes hard -> rear-end (min h_r < 0),
#                                  still forward-safe.
#
# Operating point (defaults): actual rear OVM alpha/beta = 0.40/0.32, kappa = 0.7,
# gap_r0 = 5, mild lead brake = 1.5; only the ASSUMED rear responsiveness is swept.
#
# Run from anywhere; paths are resolved from the script's own location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

OUT="examples/rear_aware/saved_results/sandwich_hocbf"
RUN="examples/rear_aware/run_sandwich.py"
PLOT="examples/rear_aware/plot_sandwich_cases.py"

# Fresh export so the registry numbers the runs deterministically run_001..004.
rm -rf "$OUT"
mkdir -p "$OUT"

# Shared scenario: aggressive ego nominal + mild lead brake + actual rear 0.40/0.32.
COMMON=(--method sandwiched_hocbf
        --rear-alpha 0.40 --rear-beta 0.32 --rear-kappa 0.7
        --gap-r0 5.0
        --output-dir "$OUT")

# run_001..004, sweeping how responsive the ego ASSUMES the rear is.
uv run python "$RUN" "${COMMON[@]}" --assumed-rear-alpha 0.00 --assumed-rear-beta 0.00 --note passive
uv run python "$RUN" "${COMMON[@]}" --assumed-rear-alpha 0.20 --assumed-rear-beta 0.16 --note under
uv run python "$RUN" "${COMMON[@]}"                                                     --note accurate
uv run python "$RUN" "${COMMON[@]}" --assumed-rear-alpha 1.00 --assumed-rear-beta 0.80 --note over

# Comparison spec (paths relative to the example dir, matching plot_sandwich_cases.py I/O).
cat > "$OUT/sandwich_cases.json" <<'JSON'
{
  "output": "saved_results/sandwich_hocbf/compare_sandwich.png",
  "runs": {
    "passive (x0.0)":  "saved_results/sandwich_hocbf/run_001",
    "under (x0.5)":    "saved_results/sandwich_hocbf/run_002",
    "accurate (x1.0)": "saved_results/sandwich_hocbf/run_003",
    "over (x2.5)":     "saved_results/sandwich_hocbf/run_004"
  }
}
JSON

# 6 stacked time series + 2 phase portraits -> saved_results/sandwich_hocbf/compare_sandwich.png
uv run python "$PLOT" "$OUT/sandwich_cases.json"

echo
echo "Exported Stage A results to $OUT/"
