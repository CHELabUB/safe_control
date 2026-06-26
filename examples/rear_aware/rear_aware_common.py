"""
Shared helpers for the rear-aware run scripts (run_three_car.py, run_ego_rear.py).

Importing this module also wires up sys.path for the sibling example packages
(double_integrator, car_following, run_registry) and configures matplotlib.

Geometry:
    Forward gap  h_f = s_lead - s_ego  - L      (ego follows lead)
    Rear gap     h_r = s_ego  - s_rear - L      (rear follows ego)
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))                  # run_registry
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'double_integrator')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'car_following')))

import matplotlib                                                  # noqa: E402
if not (os.environ.get('DISPLAY') or sys.platform == 'darwin'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402,F401

plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15,
                     'legend.fontsize': 11, 'figure.titlesize': 18})

LW = 2.4
BODY_LENGTH = 4.5
L_COMBINED = BODY_LENGTH

# Register columns (superset across scenarios) so both scripts write a consistent
# header into a shared register.csv; unused fields are left blank per row.
# The (actual vs assumed) alpha/beta columns make the interaction-model mismatch the
# register's first-class varying axis (see the interaction_accuracy sweep).
REGISTER_COLUMNS = ['scenario', 'method', 'ego_model', 'ego_target', 'v_desired',
                    'v_max', 'rear_model', 'assumed_rear_model',
                    'rear_alpha', 'rear_beta', 'assumed_rear_alpha', 'assumed_rear_beta',
                    'rear_kappa', 'rear_a_decel', 'gap_r0', 'mismatch', 'note']


def resolve(val, default):
    """CLI override if provided (not None), else the scenario-specific default."""
    return default if val is None else val


def save_run_series(save_dir, method, t_state, t_ctrl, series):
    """Save one method's time-series to ``<save_dir>/series_<method>.npz``.

    Persists every numeric array in ``series`` (plus the two time axes) so a run can be
    re-plotted later — by the per-run figure or the independent ``plot_runs.py`` — without
    re-simulating. Non-numeric / ragged entries (e.g. an empty status list) are skipped.

    Args:
        save_dir: the run folder (e.g. ``output/run_007``).
        method: short method tag used in the filename ('baseline', 'bcbf', 'hocbf', ...).
        t_state, t_ctrl: the state-length and control-length time axes.
        series: dict of per-step arrays (the simulate ``out`` dict).

    Returns:
        Path to the written ``.npz``.
    """
    arrays = {'t_state': np.asarray(t_state, dtype=float),
              't_ctrl': np.asarray(t_ctrl, dtype=float)}
    for key, val in series.items():
        arr = np.asarray(val)
        if arr.size == 0 or arr.dtype == object or arr.dtype.kind not in 'fiu':
            continue                                  # skip status lists / empty / ragged
        arrays[key] = arr.astype(float)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'series_{method}.npz')
    np.savez(path, **arrays)
    return path


def load_run_series(run_path, method=None):
    """Load a saved series ``.npz`` -> dict of arrays.

    Args:
        run_path: a run folder, or a direct path to a ``series_*.npz`` file.
        method: method tag (required when ``run_path`` is a folder).
    """
    if run_path.endswith('.npz'):
        path = run_path
    else:
        path = os.path.join(run_path, f'series_{method}.npz')
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def qp_solver_stats(statuses):
    """Summarize per-step QP solver statuses into a health-stats dict for results.json.

    Both CBF-QP controllers (CarFollowingCBF1D, RearEndBackupCBF1D) expose a
    `last_status` per step (the cvxpy `prob.status`, plus the custom 'failure' /
    'no_constraints'). This tallies them so a run records whether any QP step was
    infeasible, failed, unbounded, or only inaccurate (a solver 'warning').

    Args:
        statuses: list of per-step status strings (None entries are ignored).

    Returns:
        dict with total_steps, per-status counts, the infeasible/failure/unbounded/
        inaccurate step counts, and an overall `healthy` flag.
    """
    from collections import Counter
    statuses = [s for s in statuses if s is not None]
    counts = dict(Counter(statuses))

    def _sum(pred):
        return int(sum(c for s, c in counts.items() if pred(s)))

    n_infeasible = _sum(lambda s: 'infeasible' in s)
    n_failure = _sum(lambda s: s == 'failure')
    n_unbounded = _sum(lambda s: 'unbounded' in s)
    n_inaccurate = _sum(lambda s: s.endswith('inaccurate'))   # solver warnings
    return {
        'total_steps': len(statuses),
        'status_counts': counts,
        'num_infeasible': n_infeasible,
        'num_failure': n_failure,
        'num_unbounded': n_unbounded,
        'num_inaccurate': n_inaccurate,
        'healthy': bool(n_infeasible == 0 and n_failure == 0 and n_unbounded == 0),
    }


def registry_run(reg, cfg, force=False, override=False):
    """Dedup + allocate a run. Returns (run, ok); ok=False means skip (matched).

    On a config match:
      - override=True  -> reuse and overwrite the matched run_NNN folder in place;
      - force=True     -> recompute into a fresh run_NNN (the duplicate is kept);
      - neither        -> skip (nothing to do).
    """
    match = reg.find_match(cfg)
    if match is not None:
        if override:
            run = reg.overwrite_run(match, cfg)
            print(f"=== Overriding run: {run.name} [{run.fingerprint}] ({run.path}) ===")
            return run, True
        if not force:
            print(f"Configuration already computed as '{match.name}' at {match.path}")
            print("Nothing to do (pass --force for a new run, or --override to overwrite it).")
            return None, False
    run = reg.create_run(cfg)
    print(f"=== New run: {run.name} [{run.fingerprint}] ({run.path}) ===")
    return run, True


def save_figure(fig, save_dir, name, footnote=''):
    """Add an optional parameter footnote, save, and close the figure."""
    if footnote:
        fig.text(0.5, 0.004, footnote, ha='center', va='bottom', fontsize=9,
                 family='monospace', bbox=dict(boxstyle='round', fc='whitesmoke', ec='0.7'))
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, name)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved figure to {out_path}")
    plt.close(fig)
