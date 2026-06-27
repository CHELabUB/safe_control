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
# It then re-runs the four cases WITH the ISSf interaction-error margin (run_005..008): a
# mismatch-proportional buffer on the REAR barrier, promoted so it bites. The accurate case
# is safe and cheapest; inaccurate cases buy back the rear gap at more control intervention.
#
# Finally it writes a mirror sweep to its own folder saved_results/sandwich_bcbf_rear_sweep/
# (run_001..008): the belief is fixed and the REAL rear is swept 0.25..2.0x, giving a U-shaped
# control-intervention curve with its minimum at the accurate (1.0x) case.
#
# Operating point (validated, monotone crossing near scale ~1.75):
#   actual rear OVM alpha/beta = 0.32/0.224, kappa = 0.7
#   beta_f = 1.0, u_acc = 3, gap_r0 = 4.5, mild lead brake = 1.5
#   ISSf margin: L_inter = 4.0, residue = 0.5, rho_rear = 3e4, gamma = 2.0
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

# 8 stacked time series (incl. |u-u_nom| + effort bar) + 2 phase portraits
uv run python "$PLOT" "$OUT/sandwich_bcbf_cases.json"

# ---------------------------------------------------------------------------------------
# ISSf interaction-error margin (run_005..008): inflate the REAR barrier by a mismatch-
# proportional buffer  rear_buffer = L_inter*(|dalpha|+|dbeta|) + residue. The buffer only
# bites once the rear constraint is promoted (--rho-rear 3e4, from the secondary 1e3) and
# with a firmer class-K gain (--gamma 2.0); the interaction margin enters ONLY the rear
# barrier (front distance may shrink, never collide). Result: the accurate case stays safe
# AND cheapest (least |u-u_nom|), inaccurate cases buy back the rear gap at higher
# intervention; the extreme over-estimate is pulled from a hard rear-end to collision-free,
# though the physical sandwich keeps it from fully restoring the d_min buffer.
ISSF=(--l-inter-ratio 4.0 --l-inter-residue 0.5 --rho-rear 3e4 --gamma 2.0)
uv run python "$RUN" "${COMMON[@]}" "${ISSF[@]}" --assumed-rear-alpha 0.00 --assumed-rear-beta 0.000 --note issf_passive
uv run python "$RUN" "${COMMON[@]}" "${ISSF[@]}" --assumed-rear-alpha 0.16 --assumed-rear-beta 0.112 --note issf_under
uv run python "$RUN" "${COMMON[@]}" "${ISSF[@]}"                                                       --note issf_accurate
uv run python "$RUN" "${COMMON[@]}" "${ISSF[@]}" --assumed-rear-alpha 0.80 --assumed-rear-beta 0.560 --note issf_over

cat > "$OUT/sandwich_bcbf_issf_cases.json" <<'JSON'
{
  "output": "saved_results/sandwich_bcbf/compare_issf.png",
  "runs": {
    "passive (x0.0)":  "saved_results/sandwich_bcbf/run_005",
    "under (x0.5)":    "saved_results/sandwich_bcbf/run_006",
    "accurate (x1.0)": "saved_results/sandwich_bcbf/run_007",
    "over (x2.5)":     "saved_results/sandwich_bcbf/run_008"
  }
}
JSON
uv run python "$PLOT" "$OUT/sandwich_bcbf_issf_cases.json"

echo
echo "Exported Stage B (sandwich_bcbf) results to $OUT/"
echo "  run_001..004  baseline (no ISSf margin): over-estimate rear-ends"
echo "  run_005..008  ISSf rear margin (vary belief): accurate safe + cheapest"

# =======================================================================================
# Mirror sweep -> its own folder saved_results/sandwich_bcbf_rear_sweep/ (run_001..008):
# FIX the ego's belief (assumed rear = 0.32/0.224) and vary the REAL rear as 0.25..2.0x of it,
# with the same ISSf margin. The ISSf buffer auto-adapts to the realized mismatch
# |dalpha|+|dbeta|. scale<1 = actual more sluggish than believed = the ego OVER-estimates
# responsiveness (risky); scale=1 = accurate (config identical to the vary-belief accurate run,
# so it dedups -- no override); scale>1 = ego UNDER-estimates (conservative, safe but costly).
# Effort is U-shaped with the minimum at accurate; the extreme over-estimate (x0.25, a
# near-unbraking real rear) is unrecoverable -- rear-end and even a forward graze -- the limit of
# the reactive rear lever (an uncontrolled rear that will not brake cannot be saved by the ego).
# =======================================================================================
OUT2="examples/rear_aware/saved_results/sandwich_bcbf_rear_sweep"
rm -rf "$OUT2"; mkdir -p "$OUT2"
SWP=(--rear-kappa 0.7 --assumed-rear-alpha 0.32 --assumed-rear-beta 0.224 --output-dir "$OUT2")
declare -a SC=(0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00)
declare -a AL=(0.08 0.16 0.24 0.32 0.40 0.48 0.56 0.64)
declare -a BE=(0.056 0.112 0.168 0.224 0.280 0.336 0.392 0.448)
for i in "${!SC[@]}"; do
  uv run python "$RUN" "${COMMON[@]}" "${ISSF[@]}" "${SWP[@]}" \
    --rear-alpha "${AL[$i]}" --rear-beta "${BE[$i]}" --note "actual_x${SC[$i]}"
done

cat > "$OUT2/sandwich_bcbf_rear_sweep_cases.json" <<'JSON'
{
  "output": "saved_results/sandwich_bcbf_rear_sweep/compare_rear_sweep.png",
  "runs": {
    "x0.25 (over-est, risky)": "saved_results/sandwich_bcbf_rear_sweep/run_001",
    "x0.50 (over-est)":        "saved_results/sandwich_bcbf_rear_sweep/run_002",
    "x0.75 (over-est)":        "saved_results/sandwich_bcbf_rear_sweep/run_003",
    "x1.00 (accurate)":        "saved_results/sandwich_bcbf_rear_sweep/run_004",
    "x1.25 (under-est)":       "saved_results/sandwich_bcbf_rear_sweep/run_005",
    "x1.50 (under-est)":       "saved_results/sandwich_bcbf_rear_sweep/run_006",
    "x1.75 (under-est)":       "saved_results/sandwich_bcbf_rear_sweep/run_007",
    "x2.00 (under-est, safe)": "saved_results/sandwich_bcbf_rear_sweep/run_008"
  }
}
JSON
uv run python "$PLOT" "$OUT2/sandwich_bcbf_rear_sweep_cases.json"

echo
echo "Exported rear sweep to $OUT2/"
echo "  run_001..008  ISSf rear margin (fix belief, vary real rear 0.25..2.0x): U-shaped effort"
