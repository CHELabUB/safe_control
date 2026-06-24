"""
Interactive car-following demo: CBF-QP vs Backup CBF behind a moving lead car.

A single lead car drives a scripted stop-and-go velocity profile.  An ego car
(1D double integrator) follows it using a human car-following model (OVM default,
IDM switchable) as its nominal controller.  The ego is simulated three times on
the *identical* lead trace:

  1. nominal only          (unsafe baseline)
  2. nominal + CBF-QP      (relative-degree-1 time-headway filter)
  3. nominal + Backup CBF  (brake-to-stop backup, moving-lead barrier)

Results are overlaid in a multi-panel figure.

Usage:
    uv run python examples/car_following/run_car_following.py
    uv run python examples/car_following/run_car_following.py --nominal-model idm
"""

import os
import sys
import argparse

import numpy as np

# --- Local imports (mirrors run_backup_cbf_1d.py) -----------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'double_integrator')))

from double_integrator_1d import DoubleIntegrator1D            # noqa: E402
from car_following_models import leader_accel, lead_velocity   # noqa: E402
from car_following_cbf import CarFollowingCBF1D                 # noqa: E402
from car_following_backup_cbf import CarFollowingBackupCBF1D    # noqa: E402

import matplotlib                                               # noqa: E402
if not (os.environ.get('DISPLAY') or sys.platform == 'darwin'):
    matplotlib.use('Agg')   # headless / remote
import matplotlib.pyplot as plt                                # noqa: E402


# -----------------------------------------------------------------------
# Geometry / defaults
# -----------------------------------------------------------------------

BODY_LENGTH = 4.5                      # both cars
L_COMBINED = BODY_LENGTH               # sum of half-lengths = (4.5 + 4.5)/2

S_LEAD0 = 30.0                         # lead start position (center)
INIT_GAP = 9.0                         # initial bumper-to-bumper gap [m] (tight)
S_EGO0 = S_LEAD0 - L_COMBINED - INIT_GAP  # ego start position (center)

CONTROLLERS = ['nominal', 'cbf', 'backup']
CTRL_LABEL = {'nominal': 'Nominal only', 'cbf': 'CBF-QP', 'backup': 'Backup CBF'}
CTRL_COLOR = {'nominal': 'tab:red', 'cbf': 'tab:blue', 'backup': 'tab:green'}


# -----------------------------------------------------------------------
# Lead trace (precomputed once)
# -----------------------------------------------------------------------

def build_lead_trace(n_sim, dt, lead_params):
    """Integrate the scripted lead-car velocity profile into (s, v) arrays."""
    s = np.zeros(n_sim + 1)
    v = np.zeros(n_sim + 1)
    s[0] = S_LEAD0
    v[0] = lead_velocity(0.0, lead_params)
    for k in range(n_sim):
        v[k + 1] = lead_velocity((k + 1) * dt, lead_params)
        s[k + 1] = s[k] + v[k] * dt
    return s, v


# -----------------------------------------------------------------------
# Lead prediction for the Backup CBF horizon
# -----------------------------------------------------------------------

def _lead_prediction(s_lead, v_lead, a_decel):
    """Callable t -> predicted lead state {'x','vx'} over the backup horizon.

    Worst-case assumption: the lead may brake at constant deceleration `a_decel`
    to a stop (then hold).  This is the safe prediction for a possibly-braking
    lead; a constant-velocity prediction (a_decel=0) is over-optimistic and lets
    the ego collide when the lead actually stops hard.
    """
    def pred(t):
        if a_decel > 0.0:
            t_stop = v_lead / a_decel
            if t <= t_stop:
                s = s_lead + v_lead * t - 0.5 * a_decel * t ** 2
                v = v_lead - a_decel * t
            else:
                s = s_lead + v_lead ** 2 / (2.0 * a_decel)
                v = 0.0
        else:
            s = s_lead + v_lead * t
            v = v_lead
        return {'x': s, 'vx': v}
    return pred


# -----------------------------------------------------------------------
# Ego simulation (one controller)
# -----------------------------------------------------------------------

def simulate(mode, model, s_lead, v_lead, dt, n_sim, robot_spec,
             nom_params, cbf_kwargs, backup_kwargs):
    """Run the ego under one controller on the given lead trace.

    Returns dict of arrays: s, v, u, h (length n_sim+1 for s/v, n_sim for u/h).
    """
    robot = DoubleIntegrator1D(dt, dict(robot_spec))

    cbf = backup = None
    lead_decel_pred = 0.0
    if mode == 'cbf':
        cbf = CarFollowingCBF1D(robot_spec, **cbf_kwargs)
    elif mode == 'backup':
        bk = dict(backup_kwargs)
        lead_decel_pred = bk.pop('lead_decel_pred', 0.0)
        backup = CarFollowingBackupCBF1D(robot, dict(robot_spec), dt=dt, **bk)

    x = np.array([S_EGO0, v_lead[0]])      # start matched to lead speed
    s_arr = np.zeros(n_sim + 1)
    v_arr = np.zeros(n_sim + 1)
    u_arr = np.zeros(n_sim)
    h_arr = np.zeros(n_sim)
    s_arr[0], v_arr[0] = x[0], x[1]

    d_min = backup_kwargs['d_min']

    for k in range(n_sim):
        lead = {'x': s_lead[k], 'vx': v_lead[k], 'length': BODY_LENGTH}
        ego = {'x': x[0], 'vx': x[1], 'length': BODY_LENGTH}
        u_nom = leader_accel(model, ego, lead, nom_params)

        if mode == 'nominal':
            u = u_nom
        elif mode == 'cbf':
            u = cbf.filter(u_nom, x, s_lead[k], v_lead[k])
        else:  # backup
            backup.set_nominal_controller(lambda xx, _u=u_nom: np.array([_u]))
            backup.set_moving_obstacles(
                _lead_prediction(float(s_lead[k]), float(v_lead[k]),
                                 lead_decel_pred))
            u = float(backup.solve_control_problem(x).flat[0])

        # Position safety barrier (collision margin), common to all controllers.
        h_arr[k] = (s_lead[k] - L_COMBINED) - x[0] - d_min
        u_arr[k] = u

        x = np.array(robot.step(x, np.array([[u]]))).flatten()
        s_arr[k + 1], v_arr[k + 1] = x[0], x[1]

    return {'s': s_arr, 'v': v_arr, 'u': u_arr, 'h': h_arr}


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def make_figure(t_state, t_ctrl, s_lead, v_lead, results, d_min, save_dir):
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    ax_pos, ax_gap, ax_vel, ax_u, ax_h = axes

    # (a) positions
    ax_pos.plot(t_state, s_lead, 'k-', lw=2, label='Lead')
    for mode in CONTROLLERS:
        ax_pos.plot(t_state, results[mode]['s'], color=CTRL_COLOR[mode],
                    lw=1.6, label=f'Ego ({CTRL_LABEL[mode]})')
    ax_pos.set_ylabel('position s [m]')
    ax_pos.set_title('Positions')
    ax_pos.legend(fontsize=8, ncol=2)
    ax_pos.grid(True, alpha=0.3)

    # (b) bumper gap with d_min and collision lines
    for mode in CONTROLLERS:
        gap = (s_lead - L_COMBINED) - results[mode]['s']   # bumper-to-bumper gap
        ax_gap.plot(t_state, gap, color=CTRL_COLOR[mode], lw=1.6,
                    label=CTRL_LABEL[mode])
    ax_gap.axhline(d_min, color='gray', ls='--', lw=1.2, label=f'd_min = {d_min}')
    ax_gap.axhline(0.0, color='red', ls=':', lw=1.2, label='collision (gap=0)')
    ax_gap.set_ylabel('bumper gap [m]')
    ax_gap.set_title('Gap to lead')
    ax_gap.legend(fontsize=8, ncol=2)
    ax_gap.grid(True, alpha=0.3)

    # (c) velocities
    ax_vel.plot(t_state, v_lead, 'k-', lw=2, label='Lead')
    for mode in CONTROLLERS:
        ax_vel.plot(t_state, results[mode]['v'], color=CTRL_COLOR[mode],
                    lw=1.6, label=CTRL_LABEL[mode])
    ax_vel.axhline(0.0, color='gray', lw=0.6)
    ax_vel.set_ylabel('velocity v [m/s]')
    ax_vel.set_title('Velocities')
    ax_vel.legend(fontsize=8, ncol=2)
    ax_vel.grid(True, alpha=0.3)

    # (d) ego control u
    for mode in CONTROLLERS:
        ax_u.plot(t_ctrl, results[mode]['u'], color=CTRL_COLOR[mode],
                  lw=1.6, label=CTRL_LABEL[mode])
    ax_u.set_ylabel('control u [m/s^2]')
    ax_u.set_title('Ego acceleration')
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)

    # (e) safety barrier h
    for mode in CONTROLLERS:
        ax_h.plot(t_ctrl, results[mode]['h'], color=CTRL_COLOR[mode],
                  lw=1.6, label=CTRL_LABEL[mode])
    ax_h.axhline(0.0, color='k', lw=1.5, label='h = 0 (must stay >= 0)')
    ax_h.set_ylabel('barrier h [m]')
    ax_h.set_xlabel('time [s]')
    ax_h.set_title('Safety barrier  h = gap - d_min')
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    fig.suptitle('Car following: Nominal vs CBF-QP vs Backup CBF', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'car_following.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved figure to {out}")
    return fig


def make_animation(t_state, s_lead, results, save_dir):
    """Optional simple line animation of lead + the three ego cars."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:
        print(f"Animation unavailable: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    all_s = np.concatenate([s_lead] + [results[m]['s'] for m in CONTROLLERS])
    ax.set_xlim(all_s.min() - 5, all_s.max() + 5)
    ax.set_ylim(-1, len(CONTROLLERS) + 1)
    ax.set_yticks([len(CONTROLLERS)] + list(range(len(CONTROLLERS) - 1, -1, -1)))
    ax.set_yticklabels(['Lead'] + [CTRL_LABEL[m] for m in CONTROLLERS])
    ax.set_xlabel('position s [m]')
    ax.set_title('Car positions over time')

    lead_pt, = ax.plot([], [], 's', color='black', ms=12)
    ego_pts = {m: ax.plot([], [], 's', color=CTRL_COLOR[m], ms=12)[0]
               for m in CONTROLLERS}
    step = max(1, len(t_state) // 200)
    frames = range(0, len(t_state), step)

    def update(k):
        lead_pt.set_data([s_lead[k]], [len(CONTROLLERS)])
        for i, m in enumerate(CONTROLLERS):
            ego_pts[m].set_data([results[m]['s'][k]], [len(CONTROLLERS) - 1 - i])
        return [lead_pt] + list(ego_pts.values())

    anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=40)
    out = os.path.join(save_dir, 'car_following.gif')
    try:
        anim.save(out, writer=PillowWriter(fps=25))
        print(f"Saved animation to {out}")
    except Exception as e:
        print(f"Could not save animation: {e}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nominal-model', choices=['ovm', 'idm'], default='ovm',
                   help="ego nominal car-following model")
    p.add_argument('--tau', type=float, default=1.0, help='CBF-QP time headway [s]')
    p.add_argument('--kappa', type=float, default=1.5, help='OVM range-policy slope')
    p.add_argument('--backup-horizon', type=float, default=4.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tf', type=float, default=15.0)
    p.add_argument('--d-min', type=float, default=2.0)
    p.add_argument('--a-max', type=float, default=3.0)
    p.add_argument('--lead-decel-pred', type=float, default=4.0,
                   help="worst-case lead deceleration assumed by the Backup CBF")
    p.add_argument('--save', action='store_true', help='also save a gif animation')
    args = p.parse_args()

    dt = args.dt
    n_sim = int(round(args.tf / dt))
    t_state = np.arange(n_sim + 1) * dt
    t_ctrl = np.arange(n_sim) * dt

    robot_spec = {'model': 'DoubleIntegrator1D', 'u_max': args.a_max,
                  'a_max': args.a_max, 'v_max': 12.0, 'body_length': BODY_LENGTH}

    # Nominal car-following model parameters (OVM + IDM both keyed here).
    # Deliberately sluggish / short-headway so the *unfiltered* nominal fails to
    # brake in time when the lead stops hard -> motivates the safety filters.
    nom_params = {
        # OVM (sluggish sensitivity, short desired headway)
        'alpha': 0.5, 'beta': 0.6, 'kappa': args.kappa, 'h_st': 2.0,
        # IDM (short headway T, comfortable -> under-brakes for a hard stop)
        'v0': 12.0, 'T': 0.4, 's0': 2.0, 'b_comfort': 2.0,
        # shared
        'v_max': 12.0, 'a_max': args.a_max,
    }

    # Lead brakes somewhat harder than the ego's a_max: avoidable with *timely*
    # max braking, but a sluggish/late nominal follower cannot recover.
    lead_params = {'v_cruise': 10.0, 't_cruise': 3.0, 'a_brake': 4.0,
                   't_hold_dur': 3.0, 'a_accel': 2.0}

    cbf_kwargs = {'alpha': 1.0, 'tau': args.tau, 'd_min': args.d_min}
    backup_kwargs = {
        'backup_horizon': args.backup_horizon, 'a_max': args.a_max,
        'L': L_COMBINED, 'd_min': args.d_min,
        'alpha_fn': (lambda h: 1.0 * h),
        'alpha_terminal_fn': (lambda h: 2.0 * h),
        'eps_s': 0.1, 'eps_v': 0.1,
        'lead_decel_pred': args.lead_decel_pred,   # consumed in simulate(), not the ctor
    }

    print(f"Building lead trace (model={args.nominal_model}, dt={dt}, "
          f"tf={args.tf}, T_backup={args.backup_horizon}) ...")
    s_lead, v_lead = build_lead_trace(n_sim, dt, lead_params)

    results = {}
    for mode in CONTROLLERS:
        print(f"  Simulating ego: {CTRL_LABEL[mode]} ...")
        results[mode] = simulate(mode, args.nominal_model, s_lead, v_lead,
                                 dt, n_sim, robot_spec, nom_params,
                                 cbf_kwargs, backup_kwargs)

    # Report
    print("\nSummary (min bumper gap / min barrier h):")
    for mode in CONTROLLERS:
        gap = (s_lead - L_COMBINED) - results[mode]['s']
        min_gap = float(gap.min())
        min_h = float(results[mode]['h'].min())
        collided = min_gap <= 0.0
        print(f"  {CTRL_LABEL[mode]:12s}: min_gap={min_gap:7.3f} m  "
              f"min_h={min_h:7.3f} m  {'COLLISION' if collided else 'safe'}")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    fig = make_figure(t_state, t_ctrl, s_lead, v_lead, results, args.d_min, save_dir)
    if args.save:
        make_animation(t_state, s_lead, results, save_dir)

    try:
        plt.show(block=False)
        plt.pause(3)
    except Exception:
        pass
    plt.close('all')


if __name__ == '__main__':
    main()
