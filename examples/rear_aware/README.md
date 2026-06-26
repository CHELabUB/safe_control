# Rear-Aware Car Following

The ego (an autonomous vehicle) must stay safe with respect to a **lead** ahead
*and* avoid being **rear-ended** by an unfiltered follower behind it. The two
objectives conflict: forward safety wants the ego to brake; rear safety wants it
not to brake too hard (or to pull away). This example builds toward that combined
problem through two simpler 1D scenarios.

Builds on the `car_following` example (reuses its OVM/IDM models, the forward CBF,
and the backup-CBF machinery) and the shared `run_registry.py`.

```
Forward gap  h_f = s_lead - s_ego  - L      (ego follows the lead)
Rear gap     h_r = s_ego  - s_rear - L      (the rear follows the ego)
```
Both must stay >= 0. The rear vehicle has **no safety filter**, so it is not
guaranteed safe — the ego must account for it.

## Scenario 1 — `three_car` (motivates the problem)

`lead (scripted stop-and-go) -> ego (forward CBF) -> rear (OVM follower, no filter)`

The ego runs the forward CBF from `car_following` (safe behind the lead). The rear
is a **steady, sluggish follower**: it cruises at the same speed at its OVM
equilibrium gap (~1.6 s time-gap) with small `alpha`/`beta` (slow reaction). When
the lead stops hard, the ego brakes safely (`h_f >= 0`), but the sluggish rear
reacts too slowly and rear-ends it (`h_r < 0`) — a realistic failure driven by
*reaction lag*, not an artificially close start or a braking limit. No new
controller; it shows the failure mode. `sweep_rear_aware.py --scenario three_car`
scans responsiveness / headway / gap and reports the steady-start rear-end cases.

```bash
uv run python examples/rear_aware/run_three_car.py
# -> ego safe vs lead (min h_f > 0) but rear-ended (min h_r < 0)
```

## Scenario 2 — `ego_rear` (interaction-aware rear safety)

`ego (car-following stop nominal) + rear (OVM follower, no filter)` — no lead. The
ego's nominal brings it to a stop at a target vs a virtual stationary wall (OVM by
default, `--ego-model idm` for IDM); unfiltered, the stop rear-ends the follower.
One simulation runs the ego **three ways** and overlays them:

```bash
uv run python examples/rear_aware/run_ego_rear.py
```

**Ego objective (`--ego-target`).** By default (`stop`) the ego decelerates to the
virtual wall at `--stop-distance`. With `--ego-target speed --v-desired V` the ego instead
performs an arbitrary **target-speed transition** to `V` (no wall): there is no lead, so
the car-following lead-feedback is repurposed for speed regulation — OVM collapses to
`v̇ = (alpha+beta)(V - v)`, IDM to free-road `a(1-(v/V)^4)`. A gentler, non-zero target
generally makes rear-end safety easier to enforce (e.g. easing 9→5 m/s leaves even the
unfiltered baseline safe). Works for both `--ego-model ovm|idm`.

| controller | result | behavior |
|---|---|---|
| baseline (no filter) | `min h_r < 0` REAR-END | stops, is hit |
| backup CBF | `min h_r >= d_min` (sim-safe) | **floors it, abandons the stop** — guarantee is interaction-inconsistent |
| coupled HOCBF | `min h_r >= d_min` (sim-safe) | **brakes gently, rides the boundary** — no formal proof under input constraints |

### Backup CBF (`--method bcbf`)
The backup CBF works in simulation: it keeps `min h_r >= d_min` and the QP is
consistently feasible. The issue is a **soundness gap**, not a mechanical failure.
A backup CBF guarantees "from here, if I commit to the backup maneuver, I stay
safe." That holds for a *passive* obstacle, but the rear is **interactive** — it
reacts to what the ego actually does. The backup rollout assumes the rear reacts to
the ego's *escape*, but the ego applies the *filtered* (near-nominal) control, so
the rear reacts to that instead. The guarantee rests on a reaction the rear never
sees. It also forces an over-aggressive escape (accelerate to `v_max`, abandoning
the stop). Kept as a contrast: it illustrates what guarantee-consistent behavior
looks like when the soundness assumption happens to hold approximately.

### Coupled HOCBF (`--method hocbf`)
Model the rear **in the closed loop**: joint state `[h_r, v_ego, v_rear]` with the
rear's reaction `v̇_rear = a_r(x)` part of the dynamics, and constrain the *applied*
ego acceleration. `b = h_r - d_min` has relative degree 2, so a high-order CBF gives
a single analytic lower bound on the ego acceleration:
```
a >= a_r(x) - (alpha1 + alpha2)(v_ego - v_rear) - alpha1*alpha2*(h_r - d_min)
```
`a_r(x)` is the rear's reaction at the *actual* state (no counterfactual), so it is
interaction-consistent. The ego simply **limits its braking** so the rear can keep
pace — it rides `h_r = d_min` and still stops (a little past the target), rather
than escaping. No rollout, no QP. Gains `--hocbf-a1`, `--hocbf-a2`.

**Caveat:** this is not a formally verified CBF. Standard HOCBF validity requires
that the input constraint set (`a ∈ [a_min, a_max]`) is compatible with the
lower-bound condition everywhere on `{b >= 0}`. When the required lower bound
exceeds `a_max` (reported as `'infeasible'` in `results.json`), the condition cannot
be met and safety is not guaranteed by the theory. In practice the constraint is
rarely active at its limit, but no formal proof of forward invariance under input
saturation has been established here.

### Robustness to rear-model mismatch (`--hocbf-robust-factor`)
The HOCBF uses the ego's *assumed* rear model; the actual rear may be less
responsive. `--hocbf-robust-factor r` (in `(0,1]`) evaluates the worst-case rear
reaction with response gains scaled by `r` (the rear may brake *less* than
assumed), giving a more conservative bound. Demonstrated mismatch (actual
`alpha=beta=0.3`, HOCBF assumes `0.6/0.4`): the deterministic HOCBF (`r=1`) breaches
the margin (`min h_r ≈ 0.39 < d_min`), while `r=0.5` holds it (`min h_r ≈ 1.0`).
Assumed params are also configurable directly (`--assumed-rear-alpha/-beta/-kappa/
-a-decel`), as is the assumed model *type* (`--assumed-rear-model {ovm,idm}`,
default = actual). E.g. assuming IDM when the rear is actually OVM drops the HOCBF
margin (`min h_r` 1.0 -> 0.23 in the default case) — visible as `mismatch=True` in
`register.csv`.

### Backup-CBF details (contrast controller)
Accelerate-to-`v_max` escape flow, rollout safety `h_r(t) >= d_min`, terminal =
positive gap at the horizon. Because the rear **reacts to the ego**, the plant is
**augmented** to the joint state `[s_e, v_e, s_r, v_r]` with the rear's assumed reaction
`a_rear(x)` folded into the drift (`x_dot = f(x) + g(x) u`, `f = [v_e, 0, v_r, a_rear(x)]`,
`g = [0, 1, 0, 0]`). The backup flow and its **sensitivity matrix** `S_i = dφ_i/dx0` then
come from the framework's own `BackupCBF._integrate_backup_trajectory`
(`position_control/backup_cbf_qp.py`) — a rigorous flow STM, not an end-to-end barrier
finite difference — and the QP uses `∇h·S·g0 / ∇h·S·f0` with `∇h = [1,0,-1,0]` (no `dh/dt`,
since the rear is a state, not an exogenous obstacle). It keeps `h_r >= 0` because the
escape is robust — but, per above, the *guarantee* is unsound for an interacting rear.
(Contrast: the forward `car_following` CBF stays a 2-state ego with an analytical STM,
because the lead is exogenous and never enters the dynamics.)

The terminal (gap-at-horizon) constraint and its gain are configurable:
`--gamma-terminal G` sets the terminal class-K gain, and `--backup-terminal` /
`--no-backup-terminal` includes or drops the terminal constraint. 
There are known issue with infeasibility at the terminal state due to in sufficient
yet hard to determined preview horizon. Can choose to drop it but better supply long
enough horizon.
ego_rear **drops it by default** (`--no-backup-terminal`); pass
`--backup-terminal` to restore it. The `RearEndBackupCBF1D(use_terminal=...)` class
default keeps the terminal.

### Interaction-model accuracy matters (preliminary study)

How much does the **accuracy** of the ego's assumed rear-following model matter to the backup
CBF? Using target-speed mode (ego eases `9 → 1` m/s; actual rear OVM `α=0.4, β=0.3`), we sweep
the *assumed* rear responsiveness as a scale of the actual gains — `passive (0)` → `under (<1)`
→ `accurate (1)` → `over (>1)` — and measure both safety (`min h_r`) and performance.

Two findings:

1. **Safety is structurally robust to mismatch.** Every filtered run stays safe
   (`min h_r ≈ 2.1–2.6 ≫ d_min`) for *every* assumed model — even 2× over-estimation. With no
   lead, the ego always retains a **forward escape** (accelerate to `v_max > v_road`), so
   over-trusting the rear's braking never causes a collision: the QP re-solves each step and
   falls back on the escape. **Over-estimation is not punished here.**
2. **Mismatch shows up as performance, not a safety cliff.** A conservative assumption
   (passive / under-responsive) makes the ego hold more speed — it settles to `v_desired`
   slowly or never, with a larger steady offset and much heavier filter intervention. A more
   accurate / responsive assumption settles faster and overrides the driver less.

| assumed (scale) | `min h_r` | `t_settle` [s] | `v_offset` | intervention | outcome |
|---|---|---|---|---|---|
| baseline (no filter) | **−1.61** | 3.6 | 0.0 | 0 | **rear-end** |
| passive (0) | 2.58 | never | 0.58 | 23.0 | safe, poor perf |
| under (0.5) | 2.39 | 13.3 | 0.40 | 17.2 | safe |
| accurate (1.0) | 2.27 | 10.2 | 0.27 | 13.7 | safe |
| over (2.0) | 2.11 | 8.2 | 0.11 | 9.6 | safe |

(`t_settle` to `|v−v_desired| ≤ 0.5`; `v_offset = v_final − v_desired`; intervention `= ∫|u−u_nom| dt`.
The actual and assumed `α, β` are logged per run in `register.csv`.)

**Takeaway.** The no-lead `ego_rear` scenario demonstrates three of the four interaction
regimes — *no safety → crash; conservative model → safe but poor task performance; accurate
model → best performance* — but **not** "over-estimate → collision", because the forward escape
is always available. Punishing over-estimation needs the ego **sandwiched** by a lead in front
(the combined three-car scenario; see *Combined front + rear* below).

**Reproduce** (the sweeps default to gitignored `temp_<scenario>/` folders; the saved copies
under `saved_results/` were produced by these same commands and then moved):
```bash
# 1. find a non-degenerate operating point (baseline crashes, accurate settles);
#    sweeps ego_v0 x v_desired x v_max  ->  temp_v_regime/sweep_v_regime.csv
uv run python examples/rear_aware/sweep_rear_aware.py --scenario v_regime

# 2. at the chosen operating point, sweep the assumed/actual responsiveness scale (+ perf
#    metrics)  ->  temp_interaction_accuracy/sweep_interaction_accuracy.csv
uv run python examples/rear_aware/sweep_rear_aware.py --scenario interaction_accuracy

# 3. statistics figure (safety flat; t_settle / v_offset / intervention degrade)
uv run python examples/rear_aware/plot_accuracy_sweep.py \
    examples/rear_aware/temp_interaction_accuracy/sweep_interaction_accuracy.csv

# 4. the five representative full runs (one --method per invocation) into temp_representative/
#    operating point: ego eases 9 -> 1 m/s, actual rear OVM alpha=0.4 beta=0.3
COMMON="--tf 25 --ego-v0 9 --v-desired 1.0 --v-max 14 --v-road 12 --gap-r0 3 --d-min 1 \
        --u-acc 3 --rear-alpha 0.4 --rear-beta 0.3 --output-dir examples/rear_aware/temp_representative"
uv run python examples/rear_aware/run_ego_rear.py --method baseline $COMMON                                       # run_001
uv run python examples/rear_aware/run_ego_rear.py --method bcbf --assumed-rear-alpha 0.0 --assumed-rear-beta 0.0  $COMMON  # passive  run_002
uv run python examples/rear_aware/run_ego_rear.py --method bcbf --assumed-rear-alpha 0.2 --assumed-rear-beta 0.15 $COMMON  # under    run_003
uv run python examples/rear_aware/run_ego_rear.py --method bcbf                                                   $COMMON  # accurate run_004
uv run python examples/rear_aware/run_ego_rear.py --method bcbf --assumed-rear-alpha 0.8 --assumed-rear-beta 0.6  $COMMON  # over     run_005

# 5. behavior overlay from a JSON spec listing those five run folders (see
#    saved_results/ego_rear_interactive_compare/compare_spec.json for the exact spec)
uv run python examples/rear_aware/plot_runs.py examples/rear_aware/temp_representative/compare_spec.json
```

**Saved results:**
- [`saved_results/ego_rear_interaction_accuracy_stats/`](saved_results/ego_rear_interaction_accuracy_stats/)
  — sweep CSV + statistics figure `accuracy_sweep.png` (safety flat & safe; `t_settle` /
  `v_offset` / intervention degrade as the assumed model gets more conservative).
- [`saved_results/ego_rear_interactive_compare/`](saved_results/ego_rear_interactive_compare/)
  — five representative runs (`baseline`, `passive`, `under`, `accurate`, `over`) with the
  behavior overlay `compare_runs.png` and the `compare_spec.json` that produced it.

## Running individual methods, saved data, and comparison plots

`run_ego_rear.py` runs **one** controller per invocation, chosen with `--method`
(default `hocbf`):

```bash
uv run python examples/rear_aware/run_ego_rear.py --method baseline
uv run python examples/rear_aware/run_ego_rear.py --method bcbf
uv run python examples/rear_aware/run_ego_rear.py --method hocbf
```

The method is part of the run's config — so each is its own registry entry — and its
`config.json` records **only the parameters that method uses** (e.g. a `bcbf` run stores
the backup gains/horizon; a `hocbf` run stores the HOCBF gains; `baseline` stores neither
and no assumed-rear model). Every run **saves its time-series** to
`<run_dir>/series_<method>.npz` (`run_three_car.py` saves `series_three_car.npz`) so runs
can be re-plotted without re-simulating.

**`plot_runs.py`** is an independent comparison plotter driven by a minimal JSON spec — a
`runs` map of `{"run-name": "path"}` (each key the label, each value the run folder) plus an
optional `output` figure path (see `compare_spec.example.json`):

```bash
uv run python examples/rear_aware/plot_runs.py examples/rear_aware/compare_spec.example.json
```

```json
{
  "output": "output/compare_methods.png",
  "runs": {
    "baseline": "output/run_001",
    "backup CBF": "output/run_002",
    "HOCBF": "output/run_003"
  }
}
```

A bare `{"run-name": "path"}` object (no `runs`/`output` wrapper) is still accepted as the
runs map. Relative run paths are resolved by trying the example directory
(`examples/rear_aware`), then the spec file's directory, then the cwd — the first holding a
`series_*.npz` wins — so paths written relative to the example dir (e.g.
`saved_results/ego_rear/run_001`) or beside the spec both work. The single `series_*.npz`
in each run folder is loaded automatically. The figure has a fixed layout —
a left column of three time series (rear gap `h_r` with `h_r=0`/`d_min` dashed; ego velocity
with the target speed dashed; ego acceleration as solid actual + thin dashed nominal in the
same colour) and a right-hand **phase portrait** (ego speed `v` on the y-axis vs rear gap
`h_r` on the x-axis) overlaying the desired-speed line, vertical `h=0` and `h=d_min` lines
with the unsafe `h<d_min` region shaded red, and the rear's range-policy curve (OVM or IDM,
whichever the rear model in use is). Reference values (`d_min`, target speed, rear params)
are read from the first run's `config.json`. The output path is the spec's `output`,
overridden by `--output` when supplied, else `<spec_dir>/compare_runs.png`.

It also prints a **config-difference table** read directly from each run's `config.json`
(the full pruned config, unlike `register.csv` which keeps only a few key columns): keys
identical across all runs are hidden, and a key absent from a run's config shows as `-`.

## Parameter sweep

Finding representative cases needs tuning. `sweep_rear_aware.py` runs a grid
(metrics only, no figures) and prints the cases worth rendering, writing a summary
CSV to `output/sweep_<scenario>.csv`.

```bash
uv run python examples/rear_aware/sweep_rear_aware.py --scenario three_car   # cases that rear-end
uv run python examples/rear_aware/sweep_rear_aware.py --scenario ego_rear     # cases the filter saves
uv run python examples/rear_aware/sweep_rear_aware.py --scenario v_regime              # find an operating point (ego_v0 x v_desired x v_max)
uv run python examples/rear_aware/sweep_rear_aware.py --scenario interaction_accuracy  # assumed-vs-actual rear responsiveness (+ perf metrics)
```
The `v_regime` and `interaction_accuracy` sweeps back the *interaction-model accuracy* study
above; they default their output to a gitignored `temp_<scenario>/` folder.
Render a chosen case with the matching `run_three_car.py` / `run_ego_rear.py`
flags; each run is a registry entry under `output/run_NNN/` (dedup by config; see
`examples/README.md`). On a config that already exists, `--force` recomputes it into a
**new** `run_NNN` (keeping the duplicate) while `--override` **overwrites the existing**
`run_NNN` in place (no duplicate folder or register row). Use `--output-dir <path>` on any
of these scripts to write the results (figures, configs, `register.csv`) somewhere other
than the default `<example>/output`.

## Known limitations / TODO
- **HOCBF validity under input constraints.** The coupled HOCBF lower-bound condition
  is not always satisfiable when `a_min` is bounded (vehicle cannot decelerate
  arbitrarily hard). When the required lower bound exceeds the acceleration limit,
  the filter saturates and formal safety is not guaranteed. Infeasible steps are
  counted in `results.json`. A formally valid approach would require verifying
  compatibility of the input set with the HOCBF condition on all of `{b >= 0}`.
- **HOCBF needs a rear model.** The coupled HOCBF assumes a (bounded) model of the
  rear's reaction. `--hocbf-robust-factor` hedges against under-responsiveness, but a
  fully adversarial tailgater cannot be guaranteed against by any ego controller —
  an RSS-style "not at fault" framing is the honest fallback there.
- **Backup CBF (interaction-inconsistent guarantee).** Works in simulation (keeps
  `min h_r >= d_min`, QP consistently feasible), but the safety guarantee is
  interaction-inconsistent: it assumes the rear reacts to the ego's escape while the
  ego actually applies the filtered control. Also has a `v_max`-saturation chatter
  degeneracy worked around by keeping the escape `v_max` above road speed.
- **Combined front + rear.** The end goal is the ego *between* a lead and a follower,
  enforcing `h_f >= 0` and `h_r >= 0` together (they conflict: forward safety wants
  braking, rear safety wants gentle braking / pulling away). These scenarios are the
  building blocks; combining the forward CBF with the coupled rear HOCBF is next.

## Files
| File | Role |
|------|------|
| `run_three_car.py` | Scenario 1 (three-car): simulation, figure, registry, saved series |
| `run_ego_rear.py` | Scenario 2 (ego + rear): one `--method` (baseline/bcbf/hocbf) per run, figure, registry, saved series |
| `plot_runs.py` | independent comparison plotter (overlay saved runs via a JSON spec) |
| `plot_accuracy_sweep.py` | statistics plotter for the `interaction_accuracy` sweep CSV |
| `compare_spec.example.json` | example spec for `plot_runs.py` |
| `rear_aware_common.py` | shared constants, matplotlib, registry/figure + series save/load helpers |
| `coupled_rear_cbf.py` | `CoupledRearCBF` — interaction-consistent HOCBF (recommended) |
| `rear_backup_cbf.py` | `RearEndBackupCBF1D` — accelerate-to-escape backup CBF (contrast) |
| `rear_aware_models.py` | ego OVM-to-stop nominal + rear follower helpers |
| `sweep_rear_aware.py` | parameter sweep -> representative cases + CSV |
| `../car_following/` | reused models, forward CBF, backup-CBF base |
| `../run_registry.py` | shared run registry |
