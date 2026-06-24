# safe_control — Codebase Guide

Python library for safety-critical robot control using Control Barrier Functions (CBF) and related methods.

## Run commands

```bash
uv run python examples/test_tracking.py --model du   # basic tracking demo
uv run python examples/evade/test_evade.py           # evade scenario
uv run python examples/double_integrator/run_backup_cbf_1d.py  # 1D backup CBF demo
```

## Package layout

The project root IS the `safe_control` package (`package-dir = {"safe_control" = "."}` in pyproject.toml).

```
safe_control (root)
├── robots/           — robot dynamics models
├── position_control/ — safety controllers (CBF-QP, MPC-CBF, Backup CBF, ...)
├── shielding/        — shielding algorithms (gatekeeper, MPS)
├── envs/             — scenario environments (evade, warehouse, drifting track)
├── utils/            — animation, geometry helpers
└── examples/         — runnable demos
    ├── test_tracking.py
    ├── evade/
    ├── drift_car/
    └── double_integrator/
        ├── by_david/        — standalone DR-bCBF reference (not using framework)
        ├── double_integrator_1d.py
        ├── backup_cbf_1d_wrapper.py
        └── run_backup_cbf_1d.py
```

Import style (from any script run at repo root):
```python
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from safe_control.position_control.backup_cbf_qp import BackupCBF
```

## Robot interface

All robot classes expose a control-affine form `ẋ = f(x) + g(x)u`:

| Method | Signature | Notes |
|--------|-----------|-------|
| `f(X)` | `(n,1) → (n,1)` | drift; supports `casadi=True` |
| `g(X)` | `(n,1) → (n,m)` | control matrix |
| `step(X, U)` | `(n,1),(m,1) → (n,1)` | Euler integration with optional v_max clamp |
| `df_dx(X)` | `(n,1) → (n,n)` | Jacobian of f |
| `agent_barrier(X, obs, r, beta)` | — | continuous-time CBF h, ḣ, ∂ḣ/∂x |
| `agent_barrier_dt(x_k, u_k, obs, r)` | — | discrete-time CBF h_k, Δh, ΔΔh |
| `nominal_input(X, goal, ...)` | → `(m,1)` | PD toward goal |
| `stop(X)`, `has_stopped(X)` | — | braking helpers |

Available robots (`robots/`):

| Key | Class | State dim | Control dim |
|-----|-------|-----------|-------------|
| `si` | `SingleIntegrator2D` | 2 [x,y] | 2 [vx,vy] |
| `di` | `DoubleIntegrator2D` | 4 [x,y,vx,vy] | 2 [ax,ay] |
| `un` | `Unicycle2D` | 3 [x,y,θ] | 2 [v,ω] |
| `du` | `DynamicUnicycle2D` | 4 | 2 |
| `kb` | `KinematicBicycle2D` | 5 | 2 |
| `DynamicBicycle2D` / `DriftingCar` | — | 8 | 2 [δ̇, τ̇] |
| `Quad2D` | — | 6 | 2 |
| `Quad3D` | — | 12 | 4 |

`robot_spec` dict keys used across models: `model`, `radius`, `v_max`, `a_max`, `w_max`, `u_max`, `safety_margin`, `body_length`, `time_headway_tau`, `lane_width`.

## Position controllers (`position_control/`)

| File | Class | What it does |
|------|-------|--------------|
| `cbf_qp.py` | `CBFQP` | Standard CBF-QP: min ‖u−u_nom‖² s.t. Lf h + Lg h u ≥ −α(h) |
| `mpc_cbf.py` | `MPCCBF` | Receding-horizon MPC with CBF terminal constraint (CasADi) |
| `backup_cbf_qp.py` | `BackupCBF` | Backup CBF: rolls out backup trajectory, imposes h constraints along it |
| `optimal_decay_cbf_qp.py` | `OptimalDecayCBFQP` | Optimizes α decay rate |
| `backup_controller.py` | `LaneChangeController`, `StoppingController`, `EvadeBackupController` | Backup policies for drift car and evade scenarios |

### BackupCBF key interface

```python
cbf = BackupCBF(robot, robot_spec, dt, backup_horizon, ax)
cbf.set_nominal_controller(lambda x: u_nom)   # or
cbf.set_nominal_trajectory(x_traj, u_traj)
cbf.set_backup_controller(backup_ctrl, target)
cbf.set_environment(env)
cbf.set_moving_obstacles(obstacles_or_callable)
u_safe = cbf.solve_control_problem(state)     # returns (m,1)
```

**Internal flow**: `_integrate_backup_trajectory` → `(phi, S)` → build G,h matrices → OSQP/SCS QP.

`_alpha(h)` and `_alpha_terminal(h)` are overridable methods (linear by default).

### 1D double integrator wrapper (`examples/double_integrator/`)

`BackupCBF1D(BackupCBF)` specialises for `x=[s,v]`, `h(x)=s`:
- Exact analytical flow map φ(t) = [s₀+v₀t+u_b t²/2, v₀+u_b t]
- Exact STM S(t) = [[1,t],[0,1]] (nilpotent A, so e^{At} = I+At exactly)
- `alpha_fn`, `alpha_terminal_fn` accepted as callables
- `compare_trajectories(x0)` returns analytical vs numerical error

Invariant sets (analytical):
```
S_max = { s>0 if v≥0,  s-v²/2≥0 if v<0 }
S(T)  = { S_max conditions } AND v≥-T
```

## Shielding (`shielding/`)

- `gatekeeper.py` — Gatekeeper: backward search over nominal horizon
- `mps.py` — Model Predictive Shielding: single-step nominal horizon

## Environments (`envs/`)

| File | Class | Key attributes |
|------|-------|----------------|
| `evade_env.py` | `EvadeEnv` | `half_width`, `pocket_x_min/max`, `pocket_y_max`, `hallway_length` |
| `warehouse_env.py` | `WarehouseEnv` | `width`, `height`, `obstacles` (list of dicts with x,y,radius) |
| `drifting_env.py` | `DriftingEnv` | `track_width` |

## Obstacle format

Static: `{'x': ..., 'y': ..., 'radius': ..., 'spec': {'radius': ...}}`
Moving: `{'x':, 'y':, 'vx':, 'vy':, 'length':, 'width':, 'active': True}`
Headway CBF applies to moving obstacles ahead in the same lane when `robot_spec['time_headway_tau'] > 0`.

## Examples reference

| Script | Algorithm | Robot |
|--------|-----------|-------|
| `examples/test_tracking.py --model XX` | CBF-QP / MPC-CBF | any |
| `examples/evade/test_evade.py --algo gatekeeper\|mps` | Gatekeeper / MPS | DoubleIntegrator2D |
| `examples/drift_car/` | BackupCBF with LaneChange | DriftingCar |
| `examples/double_integrator/run_backup_cbf_1d.py` | BackupCBF1D (analytical) | 1D custom |
| `examples/double_integrator/by_david/` | DR-bCBF (disturbance-robust, standalone) | 1D |
