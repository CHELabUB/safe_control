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
import json
import argparse

import numpy as np

# --- Local imports (mirrors run_backup_cbf_1d.py) -----------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'double_integrator')))
# examples/ root holds the shared run_registry utility (reusable by any example).
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

from double_integrator_1d import DoubleIntegrator1D            # noqa: E402
from car_following_models import nominal_accel, lead_velocity  # noqa: E402
from car_following_cbf import CarFollowingCBF1D                 # noqa: E402
from car_following_backup_cbf import CarFollowingBackupCBF1D    # noqa: E402
from car_following_safe_distance import (                       # noqa: E402
    safe_distance, SmoothSafeDistance)
from run_registry import RunRegistry                            # noqa: E402

import matplotlib                                               # noqa: E402
if not (os.environ.get('DISPLAY') or sys.platform == 'darwin'):
    matplotlib.use('Agg')   # headless / remote
import matplotlib.pyplot as plt                                # noqa: E402

# Large fonts for all figures.
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 19,
})

# Line widths: profile (time-series) panels scaled 1.5x; h-v phase panel 2x.
LW_PROFILE = 1.6 * 1.5          # controller profile lines
LW_LEAD = 2.0 * 1.5             # lead reference line
LW_REF = 1.3 * 1.5              # time-series reference lines (d_min, collision, ...)
LW_PHASE = 1.6 * 2.0            # h-v trajectory lines
LW_PHASE_REF = 1.4 * 2.0        # h-v boundary lines
MS_PHASE = 5 * 2                # h-v start markers


# -----------------------------------------------------------------------
# Geometry / defaults
# -----------------------------------------------------------------------

BODY_LENGTH = 4.5                      # both cars
L_COMBINED = BODY_LENGTH               # sum of half-lengths = (4.5 + 4.5)/2

S_LEAD0 = 30.0                         # lead start position (center)

# Two scenarios contrasting the ego's braking capability relative to the lead's
# worst-case braking a_l.  The required safe distance b_hat (and hence how close
# the ego may safely follow) differs sharply between them.  init_gap is tuned so
# each starts comfortably inside its invariant set C, with margin for visibility.
#
#   ego_weak  : a_e < a_l  -> ego out-braked by lead, must keep a LARGE distance
#   ego_strong: a_e > a_l  -> ego out-brakes lead, may follow much CLOSER
#
# For the robust guarantee to hold, a_l (assumed worst-case lead braking) must be
# >= the lead's actual maximum deceleration (the scripted a_brake = 4 below); both
# scenarios therefore use a_l = 4 and contrast only the ego capability a_e.
SCENARIOS = {
    'ego_weak':   {'a_e': 2.5, 'a_l': 4.0, 'init_gap': 20.0,
                   'desc': 'ego brakes weaker than lead (a_e < a_l)'},
    'ego_strong': {'a_e': 6.0, 'a_l': 4.0, 'init_gap': 13.0,
                   'desc': 'ego brakes stronger than lead (a_e > a_l)'},
}

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
# Worst-case lead prediction for the Backup CBF horizon
# -----------------------------------------------------------------------

def _lead_prediction(s_lead, v_lead, a_decel):
    """Callable t -> worst-case predicted lead state {'x','vx'}.

    The lead is assumed to brake at constant deceleration `a_decel` (= a_l) to a
    stop and hold.  This is the robust prediction matching the maximal-invariant-
    set derivation; the actual lead follows a milder interactive profile.
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

def simulate(mode, model, s_ego0, s_lead, v_lead, dt, n_sim, robot_spec,
             nom_params, p, cbf_kwargs, backup_kwargs, smooth):
    """Run the ego under one controller on the given lead trace.

    Returns dict of arrays: s, v, u (length n_sim), b (CBF value h - b_hat), gap,
    plus 'inaccurate' = list of (step, time, status) for non-optimal QP solves.
    """
    robot = DoubleIntegrator1D(dt, dict(robot_spec))
    a_l = float(p['a_l'])

    cbf = backup = None
    if mode == 'cbf':
        cbf = CarFollowingCBF1D(robot_spec, p, **cbf_kwargs)
    elif mode == 'backup':
        backup = CarFollowingBackupCBF1D(robot, dict(robot_spec), p, dt=dt,
                                         smooth=smooth, **backup_kwargs)

    x = np.array([s_ego0, v_lead[0]])      # start matched to lead speed
    s_arr = np.zeros(n_sim + 1)
    v_arr = np.zeros(n_sim + 1)
    u_arr = np.zeros(n_sim)
    b_arr = np.zeros(n_sim)                # CBF value b = h - b_hat(v, v1)
    s_arr[0], v_arr[0] = x[0], x[1]
    inaccurate = []

    for k in range(n_sim):
        lead = {'x': s_lead[k], 'vx': v_lead[k], 'length': BODY_LENGTH}
        ego = {'x': x[0], 'vx': x[1], 'length': BODY_LENGTH}
        u_nom = nominal_accel(model, ego, lead, nom_params)

        if mode == 'nominal':
            u = u_nom
        elif mode == 'cbf':
            u = cbf.filter(u_nom, x, s_lead[k], v_lead[k])
        else:  # backup
            backup.set_nominal_controller(lambda xx, _u=u_nom: np.array([_u]))
            backup.set_moving_obstacles(
                _lead_prediction(float(s_lead[k]), float(v_lead[k]), a_l))
            u = float(backup.solve_control_problem(x).flat[0])

        ctrl = cbf if mode == 'cbf' else backup
        if ctrl is not None and getattr(ctrl, 'last_status', None) not in (
                None, 'optimal', 'no_constraints'):
            inaccurate.append((k, k * dt, ctrl.last_status))

        # CBF value b = h - b_hat(v, v1), common collision/safety metric.
        h = (s_lead[k] - L_COMBINED) - x[0]
        b_arr[k] = h - safe_distance(x[1], v_lead[k], p)

        # Non-negative ego speed: cap braking at a full stop (no reversing).
        u = max(u, -x[1] / dt)
        u_arr[k] = u

        x = np.array(robot.step(x, np.array([[u]]))).flatten()
        s_arr[k + 1], v_arr[k + 1] = x[0], x[1]

    return {'s': s_arr, 'v': v_arr, 'u': u_arr, 'b': b_arr,
            'inaccurate': inaccurate}


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def make_figure(t_state, t_ctrl, s_lead, v_lead, results, d_min, save_dir,
                phase, suffix='', title_extra='', footnote=''):
    """5 time-series panels + 1 h-v phase panel.

    phase: dict with keys tau, kappa, h_st, vbar for the phase-plane boundaries.
    footnote: compact parameter string rendered along the figure bottom.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 2, width_ratios=[1.15, 1])
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_vel = fig.add_subplot(gs[2, 0], sharex=ax_pos)
    ax_u = fig.add_subplot(gs[3, 0], sharex=ax_pos)
    ax_h = fig.add_subplot(gs[4, 0], sharex=ax_pos)
    ax_ph = fig.add_subplot(gs[:, 1])         # phase plane spans the 2nd column

    # (a) positions
    ax_pos.plot(t_state, s_lead, 'k-', lw=LW_LEAD, label='Lead')
    for mode in CONTROLLERS:
        ax_pos.plot(t_state, results[mode]['s'], color=CTRL_COLOR[mode],
                    lw=LW_PROFILE, label=f'Ego ({CTRL_LABEL[mode]})')
    ax_pos.set_ylabel('position s [m]')
    ax_pos.set_title('Positions')
    ax_pos.legend(ncol=2)
    ax_pos.grid(True, alpha=0.3)

    # (b) bumper gap with d_min and collision lines
    for mode in CONTROLLERS:
        gap = (s_lead - L_COMBINED) - results[mode]['s']   # bumper-to-bumper gap
        ax_gap.plot(t_state, gap, color=CTRL_COLOR[mode], lw=LW_PROFILE,
                    label=CTRL_LABEL[mode])
    ax_gap.axhline(d_min, color='gray', ls='--', lw=LW_REF, label=f'd_min = {d_min}')
    ax_gap.axhline(0.0, color='red', ls=':', lw=LW_REF, label='collision (gap=0)')
    ax_gap.set_ylabel('bumper gap [m]')
    ax_gap.set_title('Gap to lead')
    ax_gap.legend(ncol=2)
    ax_gap.grid(True, alpha=0.3)

    # (c) velocities
    ax_vel.plot(t_state, v_lead, 'k-', lw=LW_LEAD, label='Lead')
    for mode in CONTROLLERS:
        ax_vel.plot(t_state, results[mode]['v'], color=CTRL_COLOR[mode],
                    lw=LW_PROFILE, label=CTRL_LABEL[mode])
    ax_vel.axhline(0.0, color='gray', lw=LW_REF * 0.6)
    ax_vel.set_ylabel('velocity v [m/s]')
    ax_vel.set_title('Velocities')
    ax_vel.legend(ncol=2)
    ax_vel.grid(True, alpha=0.3)

    # (d) ego control u
    for mode in CONTROLLERS:
        ax_u.plot(t_ctrl, results[mode]['u'], color=CTRL_COLOR[mode],
                  lw=LW_PROFILE, label=CTRL_LABEL[mode])
    ax_u.set_ylabel('control u [m/s^2]')
    ax_u.set_title('Ego acceleration')
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    # (e) CBF value b = h - b_hat(v, v1)  (maximal invariant set)
    for mode in CONTROLLERS:
        ax_h.plot(t_ctrl, results[mode]['b'], color=CTRL_COLOR[mode],
                  lw=LW_PROFILE, label=CTRL_LABEL[mode])
    ax_h.axhline(0.0, color='k', lw=LW_REF * 1.2,
                 label='b = 0 (invariant boundary; CBF-QP holds b >= 0)')
    ax_h.set_ylabel('CBF  b [m]')
    ax_h.set_xlabel('time [s]')
    ax_h.set_title(r'CBF value  b = h - $\hat{b}$(v, v$_1$)'
                   '   (Backup CBF may dip < 0 but stays collision-free)')
    ax_h.legend()
    ax_h.grid(True, alpha=0.3)

    # (f) h-v phase plane (h on x-axis, v on y-axis) with reference boundaries
    tau, vbar, model = phase['tau'], phase['vbar'], phase.get('model', 'ovm')
    h_traj_all = [(s_lead - L_COMBINED) - results[m]['s'] for m in CONTROLLERS]
    h_max = max(float(h.max()) for h in h_traj_all)
    h_min = min(float(h.min()) for h in h_traj_all)
    hh = np.linspace(min(h_min, 0.0) * 1.05, h_max * 1.05, 300)
    # collision region h < 0 (cars overlapped)
    ax_ph.axvspan(hh[0], 0.0, color='red', alpha=0.07, zorder=0)
    ax_ph.axvline(0.0, color='r', ls=':', lw=LW_PHASE_REF, label='h = 0 (collision)')
    ax_ph.plot(hh, hh / tau, color='gray', ls='--', lw=LW_PHASE_REF,
               label=r'v = h/$\tau$  (target set)')

    # Desired-spacing curve at matched speed (model-dependent).
    if model == 'idm':
        v0, T, s0 = phase['v0'], phase['T'], phase['s0']
        vv = np.linspace(0.0, min(0.98 * v0, vbar), 300)
        # IDM equilibrium gap (v_dot = 0, dv = 0): h_eq = (s0 + v*T)/sqrt(1-(v/v0)^4)
        h_eq = (s0 + vv * T) / np.sqrt(np.clip(1.0 - (vv / v0) ** 4, 1e-6, None))
        ax_ph.plot(h_eq, vv, color='tab:purple', ls='-.', lw=LW_PHASE_REF,
                   label=r'IDM desired gap ($\dot v$=0, $\Delta v$=0)')
    else:
        kappa, h_st = phase['kappa'], phase['h_st']
        ax_ph.plot(hh, np.clip(kappa * (hh - h_st), 0.0, None),
                   color='tab:purple', ls='-.', lw=LW_PHASE_REF,
                   label=r'v = $\kappa$(h $-$ h$_{st}$)  (OVM desired)')

    for mode, h_traj in zip(CONTROLLERS, h_traj_all):
        ax_ph.plot(h_traj, results[mode]['v'], color=CTRL_COLOR[mode],
                   lw=LW_PHASE, label=CTRL_LABEL[mode])
        ax_ph.plot(h_traj[0], results[mode]['v'][0], 'o',
                   color=CTRL_COLOR[mode], ms=MS_PHASE)        # start marker
    ax_ph.set_xlabel('bumper gap h [m]')
    ax_ph.set_ylabel('ego speed v [m/s]')
    ax_ph.set_title('h-v phase plane (trajectories + reference boundaries)')
    ax_ph.set_xlim(min(h_min, 0.0) * 1.05 - 0.5, h_max * 1.05)
    ax_ph.set_ylim(0, vbar)
    ax_ph.legend(ncol=2)
    ax_ph.grid(True, alpha=0.3)

    fig.suptitle('Car following: Nominal vs CBF-QP vs Backup CBF'
                 + (f'  —  {title_extra}' if title_extra else ''), fontsize=13)
    if footnote:
        fig.text(0.5, 0.004, footnote, ha='center', va='bottom', fontsize=10,
                 family='monospace',
                 bbox=dict(boxstyle='round', fc='whitesmoke', ec='0.7'))
    fig.tight_layout(rect=[0, 0.025, 1, 0.99])

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'car_following{suffix}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved figure to {out}")
    return fig


def make_surface_figure(p, smooth, save_dir, suffix=''):
    """Paper Fig. 2 style: exact b_hat(v, v1) vs the fitted smooth b_hat_s >= b_hat."""
    vbar = float(p['vbar'])
    vs = np.linspace(0.0, vbar, 50)
    v1s = np.linspace(0.0, vbar, 50)
    VV, VV1 = np.meshgrid(vs, v1s, indexing='ij')
    B = np.vectorize(lambda a, b: safe_distance(a, b, p))(VV, VV1)
    Bs = np.vectorize(smooth.value)(VV, VV1)

    fig = plt.figure(figsize=(11, 4.5))
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(VV, VV1, B, cmap='viridis', alpha=0.9)
        ax1.set_title(r'Exact  $\hat{b}$(v, v$_1$)')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(VV, VV1, B, color='gray', alpha=0.35)
        ax2.plot_surface(VV, VV1, Bs, cmap='plasma', alpha=0.7)
        ax2.set_title(r'Smooth fit  $\hat{b}_s \geq \hat{b}$ (backup CBF)')
        for ax in (ax1, ax2):
            ax.set_xlabel('v [m/s]')
            ax.set_ylabel('v$_1$ [m/s]')
            ax.set_zlabel(r'$\hat{b}$ [m]')
    except Exception:
        ax1 = fig.add_subplot(1, 2, 1)
        c1 = ax1.contourf(VV, VV1, B, levels=20, cmap='viridis')
        fig.colorbar(c1, ax=ax1)
        ax1.set_title(r'Exact  $\hat{b}$(v, v$_1$)')
        ax2 = fig.add_subplot(1, 2, 2)
        c2 = ax2.contourf(VV, VV1, Bs - B, levels=20, cmap='plasma')
        fig.colorbar(c2, ax=ax2)
        ax2.set_title(r'$\hat{b}_s - \hat{b} \geq 0$ (fit margin)')
        for ax in (ax1, ax2):
            ax.set_xlabel('v [m/s]')
            ax.set_ylabel('v$_1$ [m/s]')

    fig.suptitle('Safe-distance surface (ITSC-2018 maximal invariant set)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'safe_distance_surface{suffix}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved surface figure to {out}")
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

def resolve_params(name, args):
    """Resolve every simulation parameter for a scenario.

    Shared by run_scenario() and build_run_config() so the fingerprinted config
    exactly matches what is simulated.
    """
    sc = SCENARIOS[name]
    a_e = args.a_ego_decel if args.a_ego_decel is not None else sc['a_e']
    a_l = args.a_lead_decel if args.a_lead_decel is not None else sc['a_l']
    init_gap = args.init_gap if args.init_gap is not None else sc['init_gap']

    lead_params = {'v_cruise': 10.0, 't_cruise': 3.0, 'a_brake': 4.0,
                   't_hold_dur': 3.0, 'a_accel': 2.0}
    p_safe = {'tau': args.tau, 'a_e': a_e, 'a_l': a_l, 'vbar': args.vbar}
    robot_spec = {'model': 'DoubleIntegrator1D', 'u_max': a_e,
                  'a_max': args.a_max, 'v_max': args.vbar,
                  'body_length': BODY_LENGTH}
    # Nominal car-following model parameters (OVM + IDM both keyed here).
    # Deliberately sluggish / short-headway so the *unfiltered* nominal fails to
    # brake in time when the lead stops hard -> motivates the safety filters.
    # The desired cruise speed is capped at the lead's cruise speed (not vbar):
    # otherwise the OVM keeps pushing the ego toward vbar, parking the closed
    # loop right on the b_hat switching curve and making the exact CBF chatter.
    nom_params = {
        'alpha': 0.4, 'beta': 0.4, 'kappa': args.kappa, 'h_st': 2.0,
        'v0': lead_params['v_cruise'], 'T': 0.4, 's0': 2.0, 'b_comfort': 2.0,
        'v_max': lead_params['v_cruise'], 'a_max': args.a_max, 'a_e': a_e,
    }
    return {
        'a_e': a_e, 'a_l': a_l, 'init_gap': init_gap, 'desc': sc['desc'],
        's_ego0': S_LEAD0 - L_COMBINED - init_gap,
        'robot_spec': robot_spec, 'p_safe': p_safe,
        'nom_params': nom_params, 'lead_params': lead_params,
    }


def build_run_config(args, names):
    """Assemble the fingerprinted run configuration (inputs only, no results)."""
    cfg = {
        'cli': {'nominal_model': args.nominal_model, 'tau': args.tau,
                'kappa': args.kappa, 'gamma': args.gamma, 'vbar': args.vbar,
                'a_max': args.a_max, 'backup_horizon': args.backup_horizon,
                'dt': args.dt, 'tf': args.tf, 'd_min': args.d_min},
        'scenarios': {},
    }
    for name in names:
        rp = resolve_params(name, args)
        cfg['scenarios'][name] = {
            'a_e': rp['a_e'], 'a_l': rp['a_l'], 'init_gap': rp['init_gap'],
            'p_safe': rp['p_safe'], 'nom_params': rp['nom_params'],
            'lead_params': rp['lead_params'],
        }
    return cfg


def run_scenario(name, args, save_dir):
    """Run all three controllers for one scenario and produce its figures."""
    sc = SCENARIOS[name]
    rp = resolve_params(name, args)
    a_e, a_l, init_gap, s_ego0 = rp['a_e'], rp['a_l'], rp['init_gap'], rp['s_ego0']
    robot_spec, p_safe = rp['robot_spec'], rp['p_safe']
    nom_params, lead_params = rp['nom_params'], rp['lead_params']

    dt = args.dt
    n_sim = int(round(args.tf / dt))
    t_state = np.arange(n_sim + 1) * dt
    t_ctrl = np.arange(n_sim) * dt

    cbf_kwargs = {'gamma': args.gamma, 'L': L_COMBINED}
    backup_kwargs = {
        'backup_horizon': args.backup_horizon, 'L': L_COMBINED,
        'd_min': args.d_min, 'gamma': args.gamma,
        'gamma_terminal': 2.0 * args.gamma,
    }

    print(f"\n=== Scenario '{name}': {sc['desc']} "
          f"(a_e={a_e}, a_l={a_l}, init_gap={init_gap}) ===")
    print("Fitting smooth safe-distance surface b_hat_s >= b_hat ...")
    smooth = SmoothSafeDistance(p_safe)
    print(f"  fit max-error {smooth.max_fit_error:.3f} m, "
          f"upward offset {smooth.offset:.3f} m")

    print(f"Building lead trace (model={args.nominal_model}, dt={dt}, tf={args.tf}) ...")
    s_lead, v_lead = build_lead_trace(n_sim, dt, lead_params)

    import warnings
    results = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Solution may be inaccurate.*')
        for mode in CONTROLLERS:
            print(f"  Simulating ego: {CTRL_LABEL[mode]} ...")
            results[mode] = simulate(mode, args.nominal_model, s_ego0, s_lead,
                                     v_lead, dt, n_sim, robot_spec, nom_params,
                                     p_safe, cbf_kwargs, backup_kwargs, smooth)

    print("Summary (min bumper gap / min CBF value b):")
    summary = {}
    for mode in CONTROLLERS:
        gap = (s_lead - L_COMBINED) - results[mode]['s']
        min_gap = float(gap.min())
        min_b = float(results[mode]['b'].min())
        min_v = float(results[mode]['v'].min())
        max_du = float(np.abs(np.diff(results[mode]['u'])).max())
        collided = min_gap <= 0.0
        summary[mode] = {'min_gap': min_gap, 'min_b': min_b, 'min_v': min_v,
                         'max_abs_du': max_du, 'collision': collided,
                         'n_inaccurate_qp': len(results[mode].get('inaccurate', []))}
        print(f"  {CTRL_LABEL[mode]:12s}: min_gap={min_gap:7.3f} m  "
              f"min_b={min_b:7.3f} m  min_v={min_v:6.3f} m/s  "
              f"max|du|={max_du:5.2f}  {'COLLISION' if collided else 'safe'}")

    for mode in CONTROLLERS:
        inacc = results[mode].get('inaccurate', [])
        if inacc:
            times = [f"{t:.2f}" for _, t, _ in inacc]
            warnings.warn(
                f"[{name}] {CTRL_LABEL[mode]} had {len(inacc)} non-optimal QP "
                f"solves (status={inacc[0][2]}) at t = [{', '.join(times)}] s. "
                f"Mild (usable inaccurate solutions); result still valid.")

    # Per-scenario results card (outputs + derived values only). The full input
    # configuration lives in the run-level config.json (see build_run_config); it
    # is not duplicated here.
    result_card = {
        'scenario': name,
        'description': sc['desc'],
        'inputs_ref': 'config.json (run-level)',
        'derived': {
            's_ego0': s_ego0,
            'n_sim': n_sim,
            'smooth_fit_max_error': round(smooth.max_fit_error, 4),
            'smooth_fit_offset': round(smooth.offset, 4),
        },
        'results': summary,
    }
    suffix = f'_{name}'
    os.makedirs(save_dir, exist_ok=True)
    res_path = os.path.join(save_dir, f'results{suffix}.json')
    with open(res_path, 'w') as fh:
        json.dump(result_card, fh, indent=2)
    print(f"Saved results to {res_path}")

    # Compact one-line footnote rendered on the figure itself.
    footnote = (
        f"{name}: a_e={a_e} a_l={a_l} tau={args.tau} gamma={args.gamma} "
        f"a_bar={args.a_max} vbar={args.vbar} init_gap={init_gap} "
        f"T_backup={args.backup_horizon} d_min={args.d_min} dt={dt} tf={args.tf}  |  "
        f"nominal={args.nominal_model} (alpha={nom_params['alpha']} "
        f"beta={nom_params['beta']} kappa={nom_params['kappa']} "
        f"h_st={nom_params['h_st']} v_des={nom_params['v_max']})  |  "
        f"lead: v_cruise={lead_params['v_cruise']} a_brake={lead_params['a_brake']}")

    phase = {'tau': args.tau, 'vbar': args.vbar, 'model': args.nominal_model,
             'kappa': args.kappa, 'h_st': nom_params['h_st'],
             's0': nom_params['s0'], 'T': nom_params['T'], 'v0': nom_params['v0']}
    make_figure(t_state, t_ctrl, s_lead, v_lead, results, args.d_min, save_dir,
                phase, suffix=suffix, title_extra=sc['desc'], footnote=footnote)
    make_surface_figure(p_safe, smooth, save_dir, suffix=suffix)
    if args.save:
        make_animation(t_state, s_lead, results, save_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=['ego_weak', 'ego_strong', 'both'],
                   default='both', help='braking-capability scenario(s) to run')
    p.add_argument('--nominal-model', choices=['ovm', 'idm'], default='ovm',
                   help="ego nominal car-following model")
    p.add_argument('--tau', type=float, default=1.0,
                   help='minimum time headway tau [s]')
    p.add_argument('--kappa', type=float, default=1.5, help='OVM range-policy slope')
    p.add_argument('--a-ego-decel', type=float, default=None,
                   help='override ego max deceleration a_e [m/s^2]')
    p.add_argument('--a-lead-decel', type=float, default=None,
                   help='override lead worst-case deceleration a_l [m/s^2]')
    p.add_argument('--init-gap', type=float, default=None,
                   help='override initial bumper-to-bumper gap [m]')
    p.add_argument('--gamma', type=float, default=1.0, help='CBF class-K gain')
    p.add_argument('--vbar', type=float, default=12.0, help='max speed v_bar [m/s]')
    p.add_argument('--backup-horizon', type=float, default=4.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tf', type=float, default=15.0)
    p.add_argument('--d-min', type=float, default=2.0,
                   help='rollout collision margin (backup CBF only) [m]')
    p.add_argument('--a-max', type=float, default=3.0,
                   help='ego max acceleration a_bar [m/s^2]')
    p.add_argument('--output-dir', default=None,
                   help='base output dir (default: <example>/output)')
    p.add_argument('--force', action='store_true',
                   help='recompute into a new run even if the config matches one')
    p.add_argument('--note', default='', help='optional label stored in register.csv')
    p.add_argument('--save', action='store_true', help='also save a gif animation')
    args = p.parse_args()

    names = ['ego_weak', 'ego_strong'] if args.scenario == 'both' else [args.scenario]
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output')

    # Registry: dedup by configuration, organize each run under output/run_NNN/.
    run_config = build_run_config(args, names)
    reg = RunRegistry(out_dir, key_columns=[
        'nominal_model', 'scenarios', 'tau', 'gamma', 'vbar', 'a_max',
        'backup_horizon', 'dt', 'tf', 'note'])

    match = reg.find_match(run_config)
    if match is not None and not args.force:
        print(f"\nConfiguration already computed as '{match.name}' at {match.path}")
        print("Nothing to do (pass --force to recompute into a new run).")
        return

    run = reg.create_run(run_config)
    print(f"\n=== New run: {run.name}  [{run.fingerprint}]  ({run.path}) ===")
    for name in names:
        run_scenario(name, args, run.path)
    reg.commit(run, columns={
        'nominal_model': args.nominal_model, 'scenarios': ','.join(names),
        'tau': args.tau, 'gamma': args.gamma, 'vbar': args.vbar, 'a_max': args.a_max,
        'backup_horizon': args.backup_horizon, 'dt': args.dt, 'tf': args.tf,
        'note': args.note})
    print(f"\nRegistered {run.name} in {reg.register_path}")

    try:
        plt.show(block=False)
        plt.pause(3)
    except Exception:
        pass
    plt.close('all')


if __name__ == '__main__':
    main()
