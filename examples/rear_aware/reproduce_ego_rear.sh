#!/usr/bin/env bash
#
# Reproduce the no-lead rear-aware (ego + interactive rear) results and export them to
# examples/rear_aware/saved_results/. Covers all three ego_rear result sets:
#
#   A) ego_rear/                          method comparison + ablations (run_ego_rear.py)
#        run_001 baseline (no filter)
#        run_002 bcbf  (drop terminal S_b)        run_004 bcbf  (keep terminal S_b)
#        run_003 hocbf (robust r=1)               run_005 hocbf (robust r=0)
#                                                 run_006 hocbf (robust r=0.5)
#        -> compare_baseline_bcbf_hocbf.png, compare_bcbf_terminal_constraints.png,
#           compare_hocbf_robust.png   (via plot_runs.py)
#
#   B) ego_rear_interactive_compare/      interaction-accuracy behaviour overlay (bcbf)
#        baseline + passive/under/accurate/over assumed rear, actual rear 0.4/0.3, v_max 14
#        -> compare_runs.png   (via plot_runs.py)
#
#   C) ego_rear_interaction_accuracy_stats/   accuracy statistics sweep
#        sweep_rear_aware.py --scenario interaction_accuracy  (scales 0..2)
#        -> sweep_interaction_accuracy.csv + accuracy_sweep.png (via plot_accuracy_sweep.py)
#
# In the no-lead scenario the ego can always escape forward, so the rear-aware filters are
# structurally safe under rear-model mismatch -- over-estimation costs performance, never
# safety (contrast the sandwich scenario, where it costs safety).
#
# Run from anywhere; paths are resolved from the script's own location.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

SR="examples/rear_aware/saved_results"
RUN="examples/rear_aware/run_ego_rear.py"
SWEEP="examples/rear_aware/sweep_rear_aware.py"
PLOT_RUNS="examples/rear_aware/plot_runs.py"
PLOT_SWEEP="examples/rear_aware/plot_accuracy_sweep.py"

# ===================================================================================
# A) ego_rear/ : baseline vs bcbf vs hocbf, plus terminal-constraint and robust ablations
# ===================================================================================
A="$SR/ego_rear"
rm -rf "$A"; mkdir -p "$A"
COMMON_A=(--output-dir "$A")   # default actual rear (OVM 0.6/0.4), ego_target speed

# run_001..006, in this order (the compare specs below reference these indices).
uv run python "$RUN" "${COMMON_A[@]}" --method baseline
uv run python "$RUN" "${COMMON_A[@]}" --method bcbf                               # drop terminal (default)
uv run python "$RUN" "${COMMON_A[@]}" --method hocbf --hocbf-robust-factor 1.0
uv run python "$RUN" "${COMMON_A[@]}" --method bcbf  --backup-terminal            # keep terminal S_b
uv run python "$RUN" "${COMMON_A[@]}" --method hocbf --hocbf-robust-factor 0.0
uv run python "$RUN" "${COMMON_A[@]}" --method hocbf --hocbf-robust-factor 0.5

cat > "$A/compare1.json" <<'JSON'
{
  "output": "saved_results/ego_rear/compare_baseline_bcbf_hocbf.png",
  "runs": {
    "baseline": "saved_results/ego_rear/run_001",
    "backup CBF": "saved_results/ego_rear/run_002",
    "HOCBF": "saved_results/ego_rear/run_003"
  }
}
JSON
cat > "$A/compare2.json" <<'JSON'
{
  "output": "saved_results/ego_rear/compare_bcbf_terminal_constraints.png",
  "runs": {
    "drop S_b": "saved_results/ego_rear/run_002",
    "keep S_b": "saved_results/ego_rear/run_004"
  }
}
JSON
cat > "$A/compare3.json" <<'JSON'
{
  "output": "saved_results/ego_rear/compare_hocbf_robust.png",
  "runs": {
    "HOCBF r = 1": "saved_results/ego_rear/run_003",
    "HOCBF r = 0.5": "saved_results/ego_rear/run_006",
    "HOCBF r = 0": "saved_results/ego_rear/run_005"
  }
}
JSON
uv run python "$PLOT_RUNS" "$A/compare1.json"
uv run python "$PLOT_RUNS" "$A/compare2.json"
uv run python "$PLOT_RUNS" "$A/compare3.json"

# ===================================================================================
# B) ego_rear_interactive_compare/ : assumed-rear accuracy overlay (backup CBF)
# ===================================================================================
B="$SR/ego_rear_interactive_compare"
rm -rf "$B"; mkdir -p "$B"
COMMON_B=(--output-dir "$B" --rear-alpha 0.4 --rear-beta 0.3 --v-max 14 --gap-r0 3.0)

uv run python "$RUN" "${COMMON_B[@]}" --method baseline
uv run python "$RUN" "${COMMON_B[@]}" --method bcbf --assumed-rear-alpha 0.0 --assumed-rear-beta 0.00
uv run python "$RUN" "${COMMON_B[@]}" --method bcbf --assumed-rear-alpha 0.2 --assumed-rear-beta 0.15
uv run python "$RUN" "${COMMON_B[@]}" --method bcbf                                                   # accurate
uv run python "$RUN" "${COMMON_B[@]}" --method bcbf --assumed-rear-alpha 0.8 --assumed-rear-beta 0.60

cat > "$B/compare_spec.json" <<'JSON'
{
  "output": "saved_results/ego_rear_interactive_compare/compare_runs.png",
  "runs": {
    "baseline (no filter)": "saved_results/ego_rear_interactive_compare/run_001",
    "passive (assumed 0/0)": "saved_results/ego_rear_interactive_compare/run_002",
    "under (assumed 0.2/0.15)": "saved_results/ego_rear_interactive_compare/run_003",
    "accurate (assumed=actual 0.4/0.3)": "saved_results/ego_rear_interactive_compare/run_004",
    "over (assumed 0.8/0.6)": "saved_results/ego_rear_interactive_compare/run_005"
  }
}
JSON
uv run python "$PLOT_RUNS" "$B/compare_spec.json"

# ===================================================================================
# C) ego_rear_interaction_accuracy_stats/ : accuracy statistics sweep
# ===================================================================================
C="$SR/ego_rear_interaction_accuracy_stats"
rm -rf "$C"; mkdir -p "$C"
uv run python "$SWEEP" --scenario interaction_accuracy --output-dir "$C"
uv run python "$PLOT_SWEEP" "$C/sweep_interaction_accuracy.csv"

echo
echo "Exported rear-aware (ego_rear) results to:"
echo "  $A/   (method comparison + ablations)"
echo "  $B/   (interaction-accuracy overlay)"
echo "  $C/   (accuracy statistics sweep)"
