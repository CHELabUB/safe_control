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
REGISTER_COLUMNS = ['scenario', 'ego_model', 'ego_target', 'v_desired', 'sensitivity',
                    'rear_model', 'assumed_rear_model', 'rear_kappa', 'rear_a_decel',
                    'gap_r0', 'mismatch', 'note']


def resolve(val, default):
    """CLI override if provided (not None), else the scenario-specific default."""
    return default if val is None else val


def registry_run(reg, cfg, force):
    """Dedup + allocate a run. Returns (run, ok); ok=False means skip (matched)."""
    match = reg.find_match(cfg)
    if match is not None and not force:
        print(f"Configuration already computed as '{match.name}' at {match.path}")
        print("Nothing to do (pass --force to recompute).")
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
