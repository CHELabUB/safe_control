# Interactive Car Following — CBF vs Backup CBF

A 1D longitudinal car-following demo contrasting a **standard CBF-QP** against a
**Backup CBF**, both built on the *maximal forward invariant set* of

> C. R. He and G. Orosz, **"Safety Guaranteed Connected Cruise Control"**,
> IEEE ITSC 2018. (`Papers/ITSC_2018.pdf`)

The point of the example is to show that the naive headway barrier `h - v·tau >= 0`
(the paper's *target set*) is **not** control-invariant, whereas the exact safe
distance `b_hat(v, v1)` (eqs. 13-16) is — and to compare a controller that enforces
the exact (non-smooth) set against one that brakes into a smooth over-approximation.

## Setup

### Model (paper eq. 1)
State `[h, v, v1]`:
- `h`  bumper-to-bumper distance headway
- `v`  ego (follower) speed,  `v_dot = a`  (control,  `a in [-a_e, a_bar]`)
- `v1` lead (predecessor) speed, `v1_dot = a1` (disturbance, `a1 in [-a_l, a1_bar]`)
- `h_dot = v1 - v`

The lead car is a scripted **stop-and-go** profile (cruise → hard brake to a full
stop → hold → re-accelerate). The ego's *nominal* controller is a human
car-following model — **OVM** (default) or **IDM** (`--nominal-model idm`) — that is
deliberately sluggish, so unfiltered it collides when the lead stops hard.

### Maximal invariant set (the correct CBF)
```
C = { (h, v, v1) : b(h, v, v1) = h - b_hat(v, v1) >= 0 }
```
`b_hat` is the minimal safe distance for a minimum time headway `tau`, derived from
worst-case braking of both vehicles. It is **continuous but not differentiable**
(kinks at the branch-switching curves `f1` / `f2,f3`). Two regimes:
- `a_e <= a_l` (ego out-braked): two branches, switch at `f1(v)`.
- `a_e >  a_l` (ego out-brakes lead): three branches, switches at `f2,f3`.

### Three controllers (run on the identical lead trace)
1. **Nominal only** — OVM/IDM, no safety filter (baseline; collides). It shares the
   ego's actuator limits `a in [-a_e, a_bar]` with the filters (braking bound = the
   real capability `a_e`, not the comfortable `a_bar`), so the unfiltered collision
   comes from the model's **sluggish response**, not an artificially low brake cap.
   (Verified: the nominal still collides even when allowed full `-a_e` braking.)
2. **CBF-QP** (`car_following_cbf.py`) — exact barrier `b = h - b_hat`. One
   constraint, worst-case lead braking `a1 = -a_l`:
   ```
   a <= [ gamma*b + (v1 - v) + (db_hat/dv1)*a_l ] / (db_hat/dv)
   ```
3. **Backup CBF** (`car_following_backup_cbf.py`) — rolls out an ego max-brake
   maneuver (and worst-case lead braking), requires no collision along the rollout
   and that the ego brakes into the **smooth** invariant set `{h >= b_hat_s(v,v1)}`,
   where `b_hat_s >= b_hat` is a regression-fit C^1 surface
   (`car_following_safe_distance.py`).

### Two scenarios (braking capability)
For the robust guarantee, the assumed worst-case lead braking `a_l` must be `>=` the
lead's real maximum deceleration (`a_brake = 4`); both scenarios use `a_l = 4` and
contrast only the ego capability `a_e`:

| Scenario     | `a_e` | `a_l` | regime          | expected behavior                          |
|--------------|-------|-------|-----------------|--------------------------------------------|
| `ego_weak`   | 2.5   | 4.0   | `a_e < a_l` (2-branch) | ego out-braked → must keep a **large** gap |
| `ego_strong` | 6.0   | 4.0   | `a_e > a_l` (3-branch) | ego out-brakes lead → may follow **closer** |

## Procedure / running

```bash
# both scenarios, OVM nominal (default)
uv run python examples/car_following/run_car_following.py

# one scenario, IDM nominal, custom horizon
uv run python examples/car_following/run_car_following.py --scenario ego_strong --nominal-model idm

# math self-test (b_hat continuity + b_hat_s >= b_hat)
uv run python examples/car_following/car_following_safe_distance.py
```

Key flags: `--scenario {ego_weak,ego_strong,both}`, `--nominal-model {ovm,idm}`,
`--tau`, `--gamma`, `--backup-horizon`, `--a-ego-decel`, `--a-lead-decel`,
`--init-gap`, `--vbar`, `--tf`, `--save` (gif).

### Run organization & de-duplication
Each invocation (a *run* = both scenarios for one config) is saved under
`output/run_NNN/` and indexed in `output/register.csv`. A run is fingerprinted by
its full input configuration (CLI args + resolved per-scenario params + nominal &
lead params, results excluded). If a new invocation's configuration **matches an
existing run**, it is skipped with a pointer to that run instead of recomputing.

```bash
uv run python examples/car_following/run_car_following.py                     # -> output/run_001 (ovm)
uv run python examples/car_following/run_car_following.py                     # matches run_001 -> skip
uv run python examples/car_following/run_car_following.py --nominal-model idm # -> output/run_002 (idm)
```
Run flags: `--output-dir <dir>` (default `<example>/output`), `--force` (recompute
into a new run even on a match), `--note "<label>"` (stored in `register.csv`).
The registry is a standalone, reusable utility — `run_registry.py` (`RunRegistry`).

### Outputs (in `output/run_NNN/`)
- `car_following_<scenario>.png` — left column: positions, gap, velocities, ego
  acceleration, CBF value `b`; right column: the **h-v phase portrait** (h on the
  x-axis, v on the y-axis) with the reference boundaries `h=0` (collision),
  `v=h/tau` (target set), and `v=kappa*(h-h_st)` (OVM desired equilibrium). A
  **parameter footnote** along the bottom records the exact config used.
- `config.json` — the **run-level input configuration** (CLI args + resolved
  per-scenario params + nominal & lead params); the single fingerprinted source of
  truth for reproducing the run.
- `results_<scenario>.json` — a per-scenario **results card** (derived values +
  per-controller `min_gap`, `min_b`, `min_v`, `max_abs_du`, `collision`,
  `n_inaccurate_qp`); inputs are not duplicated here (see `config.json`).
- `safe_distance_surface_<scenario>.png` — exact `b_hat` vs the smooth fit `b_hat_s`
  (paper Fig. 2 style).

## Findings

### 1. The exact invariant set works; the target set would not
The CBF-QP holds `b = h - b_hat >= 0` throughout and rides the boundary tightly
(`min_b ≈ 0.02`), i.e. it follows as closely as is provably safe. The unfiltered
nominal drives `b` (and the gap) negative — collision.

| (OVM)        | Nominal `min_gap` | CBF-QP `min_b` | Backup `min_b` | Backup `min_gap` |
|--------------|-------------------|----------------|----------------|------------------|
| `ego_weak`   | −1.61 (COLLISION) | 0.03 (rides C) | 0.90           | 2.04 ≥ d_min     |
| `ego_strong` | −4.84 (COLLISION) | 0.02 (rides C) | 0.97           | 2.09 ≥ d_min     |

### 2. Backup CBF conservativeness tracks braking capability
A backup CBF guarantees *no collision* (the ego can always brake to safety), which
is a **larger** set than `C`. So the Backup CBF can legitimately let `b` dip below 0
when the ego brakes strongly, while staying collision-free. Concretely, the
backup-safe set grows with `a_e`: the weak-braking ego must stay deep inside `C`,
the strong-braking ego may follow closer. The smooth terminal surface `b_hat_s` keeps
its constraint differentiable.

### 3. The exact CBF can chatter at the `b_hat` kink — fixed by the operating point
With an earlier tuning, the CBF-QP control chattered (`max|du| ≈ 1.0-2.1 m/s^2`)
during the **cruise phase** of `ego_weak`. Cause: `b_hat` is non-smooth, and the
closed loop parked **right on the switching curve `v1 = f1(v)`**, where the gradient
jumps (`db_hat/dv`: `1` → `4`; the `a_l` disturbance term toggles on/off). The QP
solution sits on the active constraint, so the branch flipped every step and `u`
toggled between two values.

This happened because the OVM kept pushing the ego toward `v_max = vbar = 12`, and
the equal-speed kink for `ego_weak` is at `v ≈ 11.94` — almost exactly that cap.
**Fix (no smoothing needed):** cap the follower's desired cruise speed at the lead's
cruise speed (10 m/s) instead of `vbar`. The steady state then sits inside the
`vtau` branch with margin and the control is smooth (`max|du| ≈ 0.13`). The Backup
CBF was already smooth (it uses the C^1 surface `b_hat_s`).

## TODOs

- **Smoothing the standard CBF (deferred).** If we later smooth the *exact* CBF to
  remove residual non-smoothness, the smoothing surface must hug the maximal
  invariant boundary `b_hat` tightly (small over-approximation), so that clear
  **room remains between the standard CBF and the more-conservative Backup CBF**
  (`b_hat_s`). Otherwise the two controllers collapse onto each other and the
  comparison loses meaning.
- **Tighten `b_hat_s`.** The current degree-4 polynomial fit needs a ~1.6-1.8 m
  upward offset to guarantee `b_hat_s >= b_hat`. A higher-degree / better-conditioned
  fit (or a shape-constrained fit) would reduce conservativeness while preserving the
  `>=` guarantee.
- **Backup QP conditioning.** A handful of OSQP `user_limit` solves remain in the
  soft-constrained backup QP near the rest point (the terminal STM column degenerates
  as `v -> 0`). Solutions are still usable; revisit scaling / a dedicated terminal
  handling to remove them.
- **IDM baseline does not collide.** With the current IDM nominal, the unfiltered
  car stays safe in *both* scenarios (min_gap ~0.75 / ~1.7 m), so the IDM run
  (`output/run_002`) does not demonstrate the filters rescuing a crash the way OVM
  does. Tune the IDM gains (e.g. larger desired speed / shorter headway `T`, weaker
  `b_comfort`) for a crisp baseline collision if an IDM failure case is wanted.
- **Stress the robustness margin.** The actual lead currently brakes at exactly
  `a_brake = a_l`. Add an explicit disturbance (or `a_brake` close to / above `a_l`)
  to visualize how the worst-case `a_l` assumption protects the guarantee.

## Files

| File | Role |
|------|------|
| `car_following_safe_distance.py` | exact `b_hat`, gradient, `SmoothSafeDistance` fit, self-test |
| `car_following_models.py` | OVM / IDM nominal (`nominal_accel`) + scripted lead profile |
| `car_following_cbf.py` | exact-CBF QP filter (`CarFollowingCBF1D`) |
| `car_following_backup_cbf.py` | rollout backup CBF (`CarFollowingBackupCBF1D`) |
| `run_car_following.py` | scenarios, simulation, figures, run registration |
| `../run_registry.py` | **shared** run registry (dedup by config); see `examples/README.md` |
| `Papers/ITSC_2018.pdf` | source paper |
