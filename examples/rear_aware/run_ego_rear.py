"""
Scenario 2 (ego + rear): interaction-aware rear-end avoidance.

The ego's nominal brings it to a stop at a target (OVM vs a virtual wall); a rear
OVM follower (no filter) is behind it. One controller is run per invocation, selected
with --method:

  baseline    : nominal only          -> rear-ended (h_r < 0)
  bcbf        : accelerate-to-escape   -> safe but over-aggressive (abandons the stop)
                [contrast only -- unsound for an interactive rear, see README]
  hocbf       : coupled high-order CBF -> safe and graceful (brakes gently, rides h_r=d_min)

The HOCBF models the rear *in the closed loop* and constrains the applied ego
acceleration, so the rear's reaction matches what it actually sees. Each run saves its
time-series (series_<method>.npz); overlay several with plot_runs.py.

Usage:
    uv run python examples/rear_aware/run_ego_rear.py --method hocbf
    uv run python examples/rear_aware/run_ego_rear.py --method hocbf --hocbf-robust-factor 0.5 \
        --rear-alpha 0.3 --rear-beta 0.3 --assumed-rear-alpha 0.6 --assumed-rear-beta 0.4
"""

import os
import json
import argparse

import numpy as np

from rear_aware_common import (plt, LW, BODY_LENGTH, L_COMBINED, REGISTER_COLUMNS,
                               resolve, registry_run, save_figure, save_run_series,
                               qp_solver_stats, _HERE)
from double_integrator_1d import DoubleIntegrator1D                # noqa: E402
from rear_aware_models import rear_accel, ego_stop_nominal, ego_speed_nominal  # noqa: E402
from rear_backup_cbf import RearEndBackupCBF1D                     # noqa: E402
from coupled_rear_cbf import CoupledRearCBF                        # noqa: E402
from run_registry import RunRegistry                               # noqa: E402

# Selectable controllers, with display label / colour / the results.json key prefix
# (kept stable: bcbf -> 'backup'). One method is run per invocation.
METHODS = ['baseline', 'bcbf', 'hocbf']
METHOD_STYLE = {'baseline': ('baseline', 'tab:red'),
                'bcbf': ('backup CBF', 'tab:green'),
                'hocbf': ('HOCBF (coupled)', 'tab:blue')}
RESULT_KEY = {'baseline': 'baseline', 'bcbf': 'backup', 'hocbf': 'hocbf'}


def simulate_ego_rear(cfg, dt, n_sim, mode):
    """Ego (OVM-to-stop nominal) + rear (OVM follower, no filter).

    mode in {'baseline', 'bcbf' (backup CBF), 'hocbf' (coupled)}.
    """
    L = L_COMBINED
    u_acc, v_max, v_road = cfg['u_acc'], cfg['v_max'], cfg['v_road']
    rspec = {'model': 'DoubleIntegrator1D', 'u_max': u_acc, 'a_max': u_acc,
             'v_max': v_max, 'body_length': BODY_LENGTH}
    ego = DoubleIntegrator1D(dt, dict(rspec))
    rear = DoubleIntegrator1D(dt, dict(rspec))

    cbf = None
    if mode == 'bcbf':
        cbf = RearEndBackupCBF1D(
            ego, dict(rspec), dt=dt, backup_horizon=cfg['backup_horizon'],
            u_acc=u_acc, v_max=v_max, L=L, d_min=cfg['d_min'],
            rear_model=cfg['rear_assumed_model'], rear_params=cfg['rear_assumed'],
            gamma=cfg['gamma'],
            gamma_terminal=cfg['gamma_terminal'], use_terminal=cfg['backup_terminal'])
    hocbf = None
    if mode == 'hocbf':
        hocbf = CoupledRearCBF(
            rear_model=cfg['rear_assumed_model'], rear_params=cfg['rear_assumed'],
            alpha1=cfg['hocbf_a1'], alpha2=cfg['hocbf_a2'],
            d_min=cfg['d_min'], a_e=u_acc, a_acc=u_acc,
            robust_factor=cfg.get('hocbf_robust_factor', 1.0))

    xe = np.array([cfg['s_ego0'], cfg['v_ego0']])
    xr = np.array([xe[0] - L - cfg['gap_r0'], cfg['v_ego0']])
    out = {k: np.zeros(n_sim + 1) for k in ('s_ego', 'v_ego', 's_rear', 'v_rear')}
    out['h_r'] = np.zeros(n_sim)
    out['u_ego'] = np.zeros(n_sim)
    out['u_nom'] = np.zeros(n_sim)
    out['u_rear'] = np.zeros(n_sim)
    out['qp_status'] = []                      # per-step solver/feasibility status (cbf or hocbf)
    out['s_ego'][0], out['v_ego'][0] = xe
    out['s_rear'][0], out['v_rear'][0] = xr

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Solution may be inaccurate.*')
        for k in range(n_sim):
            ego_d = {'x': xe[0], 'vx': xe[1], 'length': BODY_LENGTH}
            rear_d = {'x': xr[0], 'vx': xr[1], 'length': BODY_LENGTH}

            if cfg['ego_target'] == 'speed':
                u_nom = ego_speed_nominal(ego_d, cfg['v_desired'], cfg['ego_nom'],
                                          cfg['ego_model'])
            else:
                u_nom = ego_stop_nominal(ego_d, cfg['stop_wall_x'], cfg['ego_nom'],
                                         cfg['ego_model'])
            if cbf is not None:
                cbf.set_rear_state(xr[0], xr[1])
                cbf.set_nominal_controller(lambda x, _u=u_nom: np.array([_u]))
                u_e = float(cbf.solve_control_problem(xe).flat[0])
                out['qp_status'].append(cbf.last_status)
            elif hocbf is not None:
                h_r = (xe[0] - L) - xr[0]
                u_e = hocbf.filter(u_nom, h_r, xe[1], xr[1])
                out['qp_status'].append(hocbf.last_status)
            else:
                u_e = u_nom
            u_e = max(u_e, -xe[1] / dt)

            u_r = rear_accel(rear_d, ego_d, cfg['rear_model'], cfg['rear_actual'])
            u_r = max(u_r, -xr[1] / dt)

            out['h_r'][k] = (xe[0] - L) - xr[0]
            out['u_ego'][k] = u_e
            out['u_nom'][k] = u_nom
            out['u_rear'][k] = u_r

            xe = np.array(ego.step(xe, np.array([[u_e]]))).flatten()
            xe[1] = min(max(xe[1], 0.0), v_max)         # ego may escape above road speed
            xr = np.array(rear.step(xr, np.array([[u_r]]))).flatten()
            xr[1] = min(max(xr[1], 0.0), v_road)        # rear capped at road speed
            out['s_ego'][k + 1], out['v_ego'][k + 1] = xe
            out['s_rear'][k + 1], out['v_rear'][k + 1] = xr
    return out


def make_figure(method, out, cfg, t_state, t_ctrl, save_dir, footnote):
    """Plot the single method's run (ego solid, rear dashed)."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    axp, axg, axv, axu = axes
    speed_mode = cfg['ego_target'] == 'speed'
    tag, c = METHOD_STYLE[method]

    axp.plot(t_state, out['s_ego'], color=c, lw=LW, label=f'Ego ({tag})')
    axp.plot(t_state, out['s_rear'], color=c, lw=LW, ls='--', label='Rear')
    if not speed_mode:
        axp.axhline(cfg['stop_wall_x'], color='k', ls=':', lw=2, label='stop target')
    axp.set_ylabel('position [m]'); axp.set_title('Positions (ego solid, rear dashed)')
    axp.legend(ncol=2, fontsize=9); axp.grid(alpha=0.3)

    axg.plot(t_ctrl, out['h_r'], color=c, lw=LW, label=f'{tag}')
    axg.axhline(0.0, color='red', ls=':', lw=2, label='collision (h_r=0)')
    axg.axhline(cfg['d_min'], color='gray', ls='--', lw=1.5, label=f"d_min={cfg['d_min']}")
    axg.set_ylabel('rear gap h_r [m]'); axg.set_title('Rear gap h_r')
    axg.legend(ncol=2); axg.grid(alpha=0.3)

    axv.plot(t_state, out['v_ego'], color=c, lw=LW, label=f'Ego ({tag})')
    axv.plot(t_state, out['v_rear'], color=c, lw=LW, ls='--', label='Rear')
    if speed_mode:
        axv.axhline(cfg['v_desired'], color='k', ls=':', lw=2, label='target speed')
    axv.set_ylabel('velocity [m/s]'); axv.set_title('Velocities (ego solid, rear dashed)')
    axv.legend(ncol=2, fontsize=9); axv.grid(alpha=0.3)

    axu.plot(t_ctrl, out['u_nom'], color='gray', lw=LW * 0.8, ls=':',
             label='ego nominal (unfiltered)')
    axu.plot(t_ctrl, out['u_ego'], color=c, lw=LW, label=f'ego ({tag})')
    axu.set_ylabel('ego accel [m/s^2]'); axu.set_xlabel('time [s]'); axu.set_title('Ego control')
    axu.legend(fontsize=9); axu.grid(alpha=0.3)

    fig.suptitle(f"Ego + rear ({tag})")
    save_figure(fig, save_dir, 'ego_rear.png', footnote)


def build_cfg(args, method=None):
    """Config for one method, holding only the parameters that method actually uses.

    The cfg is both saved as config.json and fingerprinted for dedup, so pruning to the
    relevant params keeps each method's record clean (e.g. a hocbf run is not
    distinguished by an unused backup gain).
    """
    if method is None:
        method = getattr(args, 'method', 'hocbf')
    # Road/desired speed for the OVM models, decoupled from the ego escape speed.
    v_road = args.v_road
    ego_nom = {'alpha': 0.6, 'beta': 0.3, 'kappa': 0.5, 'h_st': 2.0,
               'v0': v_road, 'v_max': v_road,
               'a_max': args.u_acc, 'a_e': args.u_acc,
               'T': 0.4, 's0': 2.0, 'b_comfort': 2.0}
    rear_actual = {'alpha': resolve(args.rear_alpha, 0.6),
                   'beta': resolve(args.rear_beta, 0.4),
                   'kappa': resolve(args.rear_kappa, 1.0),
                   'h_st': resolve(args.rear_hst, 1.0),
                   'v0': v_road, 'v_max': v_road,
                   'a_max': resolve(args.rear_a_max, 2.0),
                   'a_e': resolve(args.rear_a_decel, 2.0),
                   'T': resolve(args.rear_T, 1.0), 's0': 2.0,
                   'b_comfort': resolve(args.rear_b, 2.0)}

    # Common: sim setup + ego nominal + the *actual* rear (used by every method).
    cfg = {
        'scenario': 'ego_rear', 'method': method, 'ego_model': args.ego_model,
        'ego_target': args.ego_target,
        's_ego0': 0.0, 'v_ego0': args.ego_v0, 'gap_r0': resolve(args.gap_r0, 3.0),
        'd_min': args.d_min, 'u_acc': args.u_acc, 'v_max': args.v_max, 'v_road': v_road,
        'ego_nom': ego_nom, 'rear_model': args.rear_model, 'rear_actual': rear_actual,
    }
    # Only the nominal objective actually in effect.
    if args.ego_target == 'speed':
        cfg['v_desired'] = resolve(args.v_desired, v_road)
    else:
        cfg['stop_wall_x'] = args.stop_distance

    # The assumed rear model is only consulted by the filters (baseline has no filter).
    if method in ('bcbf', 'hocbf'):
        rear_assumed = dict(rear_actual)
        for key, val in [('kappa', args.assumed_rear_kappa), ('a_e', args.assumed_rear_a_decel),
                         ('alpha', args.assumed_rear_alpha), ('beta', args.assumed_rear_beta)]:
            if val is not None:
                rear_assumed[key] = val
        cfg['rear_assumed_model'] = resolve(args.assumed_rear_model, args.rear_model)
        cfg['rear_assumed'] = rear_assumed
    if method == 'bcbf':
        cfg.update({'gamma': args.gamma, 'gamma_terminal': args.gamma_terminal,
                    'backup_terminal': args.backup_terminal,
                    'backup_horizon': args.backup_horizon})
    if method == 'hocbf':
        cfg.update({'hocbf_a1': args.hocbf_a1, 'hocbf_a2': args.hocbf_a2,
                    'hocbf_robust_factor': args.hocbf_robust_factor})
    return cfg


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', default='baseline', choices=['baseline', 'bcbf', 'hocbf'],
                   help="controller to run (one per invocation; default: baseline). "
                        "'bcbf' = backup CBF, 'hocbf' = coupled HOCBF. Saved as "
                        "series_<method>.npz; compare runs with plot_runs.py.")
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tf', type=float, default=12.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--gamma-terminal', type=float, default=2.0,
                   help='backup-CBF terminal (gap-at-horizon) class-K gain')
    p.add_argument('--backup-terminal', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='include the backup-CBF terminal constraint (--backup-terminal) '
                        'or drop it (default for ego_rear)')
    p.add_argument('--ego-model', choices=['ovm', 'idm'], default='ovm',
                   help='ego nominal car-following model (used to stop at the target)')
    p.add_argument('--ego-target', choices=['speed', 'stop'], default='speed',
                   help="ego nominal objective: 'speed' regulates to --v-desired (no "
                        "wall); 'stop' decelerates to the virtual wall at --stop-distance")
    p.add_argument('--v-desired', type=float, default=1.0,
                   help='ego target cruise speed for --ego-target speed [m/s] '
                        '(default: --v-road)')
    p.add_argument('--ego-v0', type=float, default=9.0, help='ego initial speed [m/s]')
    p.add_argument('--stop-distance', type=float, default=18.0,
                   help='ego stop target (virtual wall) ahead of start [m]')
    p.add_argument('--gap-r0', type=float, default=None, help='initial rear-ego gap')
    p.add_argument('--d-min', type=float, default=1.0, help='rear collision margin [m]')
    p.add_argument('--u-acc', type=float, default=3.0, help='ego accel/brake magnitude [m/s^2]')
    p.add_argument('--v-road', type=float, default=12.0, help='road/desired speed [m/s]')
    p.add_argument('--v-max', type=float, default=15.0, help='ego escape speed (> v_road)')
    p.add_argument('--backup-horizon', type=float, default=5.0)
    p.add_argument('--hocbf-a1', type=float, default=1.0, help='HOCBF gain alpha1')
    p.add_argument('--hocbf-a2', type=float, default=1.0, help='HOCBF gain alpha2')
    p.add_argument('--hocbf-robust-factor', type=float, default=1.0,
                   help='HOCBF robustness: <1 assumes the rear is that fraction as '
                        'responsive (worst-case); 1.0 = deterministic')
    # Rear follower params (actual) and what the filter assumes.
    p.add_argument('--rear-model', choices=['ovm', 'idm'], default='ovm',
                   help='actual rear car-following model')
    p.add_argument('--assumed-rear-model', choices=['ovm', 'idm'], default=None,
                   help='model the filter assumes for the rear (default = actual)')
    p.add_argument('--rear-alpha', type=float, default=None)
    p.add_argument('--rear-beta', type=float, default=None)
    p.add_argument('--rear-kappa', type=float, default=None)
    p.add_argument('--rear-hst', type=float, default=None)
    p.add_argument('--rear-a-max', type=float, default=None)
    p.add_argument('--rear-a-decel', type=float, default=None)
    p.add_argument('--rear-T', type=float, default=None)
    p.add_argument('--rear-b', type=float, default=None)
    p.add_argument('--assumed-rear-alpha', type=float, default=None)
    p.add_argument('--assumed-rear-beta', type=float, default=None)
    p.add_argument('--assumed-rear-kappa', type=float, default=None)
    p.add_argument('--assumed-rear-a-decel', type=float, default=None)
    p.add_argument('--output-dir', default=None,
                   help='results directory (default: <example>/output); runs are '
                        'saved as <output-dir>/run_NNN/')
    p.add_argument('--force', action='store_true',
                   help='recompute a matching config into a fresh run_NNN (keeps the duplicate)')
    p.add_argument('--override', action='store_true',
                   help='overwrite the matching run_NNN in place instead of creating a new one')
    p.add_argument('--note', default='')
    return p


def main():
    args = build_parser().parse_args()
    out_dir = args.output_dir or os.path.join(_HERE, 'output')
    dt = args.dt
    n_sim = int(round(args.tf / dt))
    t_state = np.arange(n_sim + 1) * dt
    t_ctrl = np.arange(n_sim) * dt

    method = args.method
    cfg = build_cfg(args, method)
    reg = RunRegistry(out_dir, key_columns=REGISTER_COLUMNS)
    run, ok = registry_run(reg, cfg, args.force, args.override)
    if not ok:
        return

    # Run the single selected method and persist its time-series for re-plotting.
    out = simulate_ego_rear(cfg, dt, n_sim, method)
    save_run_series(run.path, method, t_state, t_ctrl, out)

    is_filter = method in ('bcbf', 'hocbf')
    mismatch = bool(is_filter and (cfg['rear_assumed'] != cfg['rear_actual']
                                   or cfg['rear_assumed_model'] != cfg['rear_model']))
    if cfg['ego_target'] == 'speed':
        print(f"  ego target: regulate to v_desired = {cfg['v_desired']:.1f} m/s (no wall)")
    else:
        print(f"  ego target: stop at wall x = {cfg['stop_wall_x']:.1f} m")

    mhr = float(out['h_r'].min())
    vmx = float(out['v_ego'].max())
    results = {'method': method, 'param_mismatch': mismatch,
               f"{RESULT_KEY[method]}_min_h_r": mhr, f"{RESULT_KEY[method]}_peak_v": vmx}
    if method == 'baseline':
        results['baseline_rear_end'] = bool(mhr <= 0)
        print(f"  baseline  : min h_r = {mhr:7.3f} m  ({'REAR-END' if mhr <= 0 else 'safe'})")
    elif method == 'bcbf':
        print(f"  backup CBF: min h_r = {mhr:7.3f} m  (peak v={vmx:.1f})")
        backup_qp = qp_solver_stats(out['qp_status'])
        results['backup_qp'] = backup_qp
        print(f"  backup QP : {backup_qp['total_steps']} steps, "
              f"infeasible={backup_qp['num_infeasible']}, failure={backup_qp['num_failure']}, "
              f"unbounded={backup_qp['num_unbounded']}, inaccurate={backup_qp['num_inaccurate']} "
              f"({'healthy' if backup_qp['healthy'] else 'UNHEALTHY'})")
    elif method == 'hocbf':
        results['hocbf_safe'] = bool(mhr > 0)
        print(f"  HOCBF     : min h_r = {mhr:7.3f} m  (peak v={vmx:.1f}, "
              f"rides boundary)  [a1={cfg['hocbf_a1']},a2={cfg['hocbf_a2']}]")
        # HOCBF is analytic (no QP); 'infeasible' here = required accel exceeded a_acc.
        hocbf_feas = qp_solver_stats(out['qp_status'])
        results['hocbf_feasibility'] = hocbf_feas
        print(f"  HOCBF feas: {hocbf_feas['total_steps']} steps, "
              f"infeasible={hocbf_feas['num_infeasible']} "
              f"({'healthy' if hocbf_feas['healthy'] else 'UNHEALTHY'})")
    with open(os.path.join(run.path, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    rp = cfg['rear_actual']
    rear_tag = cfg['rear_model']
    if is_filter and cfg['rear_assumed_model'] != cfg['rear_model']:
        rear_tag += f"/assumed={cfg['rear_assumed_model']}"
    ego_target_tag = (f"speed(v_des={cfg['v_desired']})" if cfg['ego_target'] == 'speed'
                      else f"stop={args.stop_distance}")
    parts = [f"ego_rear[{method}]: ego={cfg['ego_model']}(u_acc={args.u_acc},{ego_target_tag})",
             f"rear={rear_tag}(kappa={rp['kappa']},a_e={rp['a_e']})",
             f"gap_r0={cfg['gap_r0']} d_min={args.d_min}"]
    if method == 'bcbf':
        parts.append(f"backup(g={cfg['gamma']},gT={cfg['gamma_terminal']},"
                     f"term={cfg['backup_terminal']})")
    if method == 'hocbf':
        parts.append(f"hocbf(a1={cfg['hocbf_a1']},a2={cfg['hocbf_a2']},"
                     f"rf={cfg['hocbf_robust_factor']})")
    if is_filter:
        parts.append(f"mismatch={mismatch}")
    footnote = ' '.join(parts)
    make_figure(method, out, cfg, t_state, t_ctrl, run.path, footnote)
    reg.commit(run, columns={'scenario': 'ego_rear', 'method': method,
                             'ego_model': cfg['ego_model'],
                             'ego_target': cfg['ego_target'],
                             'v_desired': cfg.get('v_desired', ''),
                             'rear_model': cfg['rear_model'],
                             'assumed_rear_model': cfg.get('rear_assumed_model', ''),
                             'rear_kappa': rp['kappa'], 'rear_a_decel': rp['a_e'],
                             'gap_r0': cfg['gap_r0'], 'mismatch': mismatch,
                             'note': args.note})
    print(f"Registered {run.name}")


if __name__ == '__main__':
    main()
