"""
Scenario 1 (three-car): motivates the rear-aware problem.

    lead (scripted stop-and-go) -> ego (forward CBF w.r.t. lead) ->
    rear (steady, sluggish OVM follower, NO filter)

The ego stays safe behind the lead (h_f >= 0), but its hard braking rear-ends the
sluggish follower (h_r < 0) -- a realistic failure from reaction lag, not an
artificially close start. No new controller; it shows the failure mode.

Usage:
    uv run python examples/rear_aware/run_three_car.py
"""

import os
import json
import argparse

import numpy as np

from rear_aware_common import (plt, LW, BODY_LENGTH, L_COMBINED, REGISTER_COLUMNS,
                               resolve, registry_run, save_figure, save_run_series,
                               qp_solver_stats, _HERE)
from double_integrator_1d import DoubleIntegrator1D                # noqa: E402
from car_following_models import nominal_accel, lead_velocity      # noqa: E402
from car_following_cbf import CarFollowingCBF1D                    # noqa: E402
from rear_aware_models import rear_accel                           # noqa: E402
from run_registry import RunRegistry                               # noqa: E402


def build_lead_trace(s0, n_sim, dt, lead_params):
    s = np.zeros(n_sim + 1)
    v = np.zeros(n_sim + 1)
    s[0] = s0
    v[0] = lead_velocity(0.0, lead_params)
    for k in range(n_sim):
        v[k + 1] = lead_velocity((k + 1) * dt, lead_params)
        s[k + 1] = s[k] + v[k] * dt
    return s, v


def simulate_three_car(cfg, dt, n_sim):
    """Lead (scripted) -> ego (forward CBF) -> rear (OVM, no filter)."""
    L = L_COMBINED
    lead_params = cfg['lead']
    s_lead, v_lead = build_lead_trace(cfg['s_lead0'], n_sim, dt, lead_params)

    p_safe = cfg['p_safe']
    ego_robot_spec = {'model': 'DoubleIntegrator1D', 'u_max': p_safe['a_e'],
                      'a_max': cfg['ego_a_max'], 'v_max': p_safe['vbar'],
                      'body_length': BODY_LENGTH}
    cbf = CarFollowingCBF1D(ego_robot_spec, p_safe, gamma=cfg['gamma'], L=L)
    ego = DoubleIntegrator1D(dt, dict(ego_robot_spec))
    ego_nom = cfg['ego_nom']

    rear = DoubleIntegrator1D(dt, dict(ego_robot_spec))
    rear_params = cfg['rear_params']

    xe = np.array([s_lead[0] - L - cfg['gap_f0'], v_lead[0]])
    xr = np.array([xe[0] - L - cfg['gap_r0'], v_lead[0]])

    out = {k: np.zeros(n_sim + 1) for k in ('s_ego', 'v_ego', 's_rear', 'v_rear')}
    out['h_f'] = np.zeros(n_sim)
    out['h_r'] = np.zeros(n_sim)
    out['u_ego'] = np.zeros(n_sim)
    out['u_rear'] = np.zeros(n_sim)
    out['qp_status'] = []                       # per-step forward CBF-QP solver status
    out['s_ego'][0], out['v_ego'][0] = xe
    out['s_rear'][0], out['v_rear'][0] = xr

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Solution may be inaccurate.*')
        for k in range(n_sim):
            lead = {'x': s_lead[k], 'vx': v_lead[k], 'length': BODY_LENGTH}
            ego_d = {'x': xe[0], 'vx': xe[1], 'length': BODY_LENGTH}
            rear_d = {'x': xr[0], 'vx': xr[1], 'length': BODY_LENGTH}

            u_e_nom = nominal_accel(cfg['ego_model'], ego_d, lead, ego_nom)
            u_e = cbf.filter(u_e_nom, xe, s_lead[k], v_lead[k])
            out['qp_status'].append(cbf.last_status)
            u_e = max(u_e, -xe[1] / dt)

            u_r = rear_accel(rear_d, ego_d, cfg['rear_model'], rear_params)
            u_r = max(u_r, -xr[1] / dt)

            out['h_f'][k] = (s_lead[k] - L) - xe[0]
            out['h_r'][k] = (xe[0] - L) - xr[0]
            out['u_ego'][k] = u_e
            out['u_rear'][k] = u_r

            xe = np.array(ego.step(xe, np.array([[u_e]]))).flatten()
            xe[1] = max(xe[1], 0.0)
            xr = np.array(rear.step(xr, np.array([[u_r]]))).flatten()
            xr[1] = max(xr[1], 0.0)
            out['s_ego'][k + 1], out['v_ego'][k + 1] = xe
            out['s_rear'][k + 1], out['v_rear'][k + 1] = xr

    out['s_lead'], out['v_lead'] = s_lead, v_lead
    return out


def make_figure(out, t_state, t_ctrl, save_dir, footnote):
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    axp, axg, axv, axu = axes

    axp.plot(t_state, out['s_lead'], 'k-', lw=LW, label='Lead')
    axp.plot(t_state, out['s_ego'], color='tab:blue', lw=LW, label='Ego (CBF vs lead)')
    axp.plot(t_state, out['s_rear'], color='tab:orange', lw=LW, label='Rear (OVM, no filter)')
    axp.set_ylabel('position [m]'); axp.set_title('Positions'); axp.legend(); axp.grid(alpha=0.3)

    axg.plot(t_ctrl, out['h_f'], color='tab:blue', lw=LW, label='h_f  (ego - lead)')
    axg.plot(t_ctrl, out['h_r'], color='tab:orange', lw=LW, label='h_r  (rear - ego)')
    axg.axhline(0.0, color='red', ls=':', lw=2, label='collision (h=0)')
    axg.set_ylabel('gap [m]'); axg.set_title('Forward gap h_f (safe) vs rear gap h_r (rear-end)')
    axg.legend(); axg.grid(alpha=0.3)

    axv.plot(t_state, out['v_lead'], 'k-', lw=LW, label='Lead')
    axv.plot(t_state, out['v_ego'], color='tab:blue', lw=LW, label='Ego')
    axv.plot(t_state, out['v_rear'], color='tab:orange', lw=LW, label='Rear')
    axv.set_ylabel('velocity [m/s]'); axv.set_title('Velocities'); axv.legend(); axv.grid(alpha=0.3)

    axu.plot(t_ctrl, out['u_ego'], color='tab:blue', lw=LW, label='Ego')
    axu.plot(t_ctrl, out['u_rear'], color='tab:orange', lw=LW, label='Rear')
    axu.set_ylabel('accel [m/s^2]'); axu.set_xlabel('time [s]'); axu.set_title('Controls')
    axu.legend(); axu.grid(alpha=0.3)

    fig.suptitle('Three-car: ego safe vs lead, but rear-ended (no rear-aware filter)')
    save_figure(fig, save_dir, 'three_car.png', footnote)


def build_cfg(args):
    lead_params = {'v_cruise': 10.0, 't_cruise': 3.0, 'a_brake': 4.0,
                   't_hold_dur': 3.0, 'a_accel': 2.0}
    ego_nom = {'alpha': 0.4, 'beta': 0.4, 'kappa': 1.5, 'h_st': 2.0,
               'v0': lead_params['v_cruise'], 'v_max': lead_params['v_cruise'],
               'a_max': args.ego_a_max, 'a_e': args.ego_a_decel,
               'T': 0.4, 's0': 2.0, 'b_comfort': 2.0}
    # Rear follower: a *sluggish* driver (small alpha/beta) cruising at a steady
    # ~1.6 s gap (its OVM equilibrium), rear-ended by slow reaction.
    rear_params = {'alpha': resolve(args.rear_alpha, 0.25),
                   'beta': resolve(args.rear_beta, 0.25),
                   'kappa': resolve(args.rear_kappa, 0.7),
                   'h_st': resolve(args.rear_hst, 2.0),
                   'v0': lead_params['v_cruise'], 'v_max': lead_params['v_cruise'],
                   'a_max': resolve(args.rear_a_max, 2.0),
                   'a_e': resolve(args.rear_a_decel, 2.5),
                   'T': resolve(args.rear_T, 1.0), 's0': 2.0,
                   'b_comfort': resolve(args.rear_b, 2.0)}
    gap_r0 = resolve(args.gap_r0, 18.0)      # ~= OVM equilibrium gap (steady start)
    return {
        'scenario': 'three_car', 'ego_model': args.ego_model,
        's_lead0': 60.0, 'gap_f0': args.gap_f0, 'gap_r0': gap_r0,
        'gamma': args.gamma, 'ego_a_max': args.ego_a_max,
        'p_safe': {'tau': args.tau, 'a_e': args.ego_a_decel,
                   'a_l': args.a_lead_decel, 'vbar': 12.0},
        'lead': lead_params, 'ego_nom': ego_nom,
        'rear_model': args.rear_model, 'rear_params': rear_params,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tf', type=float, default=14.0)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--ego-model', choices=['ovm', 'idm'], default='ovm',
                   help='ego nominal car-following model')
    p.add_argument('--ego-a-max', type=float, default=3.0)
    p.add_argument('--ego-a-decel', type=float, default=4.0)
    p.add_argument('--a-lead-decel', type=float, default=4.0)
    p.add_argument('--gap-f0', type=float, default=16.0, help='initial ego-lead gap')
    p.add_argument('--gap-r0', type=float, default=None,
                   help='initial rear-ego gap (default ~ OVM equilibrium)')
    p.add_argument('--rear-model', choices=['ovm', 'idm'], default='ovm')
    p.add_argument('--rear-alpha', type=float, default=None)
    p.add_argument('--rear-beta', type=float, default=None)
    p.add_argument('--rear-kappa', type=float, default=None)
    p.add_argument('--rear-hst', type=float, default=None)
    p.add_argument('--rear-a-max', type=float, default=None)
    p.add_argument('--rear-a-decel', type=float, default=None)
    p.add_argument('--rear-T', type=float, default=None)
    p.add_argument('--rear-b', type=float, default=None)
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

    cfg = build_cfg(args)
    reg = RunRegistry(out_dir, key_columns=REGISTER_COLUMNS)
    run, ok = registry_run(reg, cfg, args.force, args.override)
    if not ok:
        return

    out = simulate_three_car(cfg, dt, n_sim)
    save_run_series(run.path, 'three_car', t_state, t_ctrl, out)
    min_hf, min_hr = float(out['h_f'].min()), float(out['h_r'].min())
    print(f"  min h_f = {min_hf:.3f} m ({'SAFE' if min_hf > 0 else 'collide'} vs lead)")
    print(f"  min h_r = {min_hr:.3f} m ({'rear-end!' if min_hr <= 0 else 'safe'} from rear)")

    ego_cbf_qp = qp_solver_stats(out['qp_status'])
    print(f"  ego CBF-QP: {ego_cbf_qp['total_steps']} steps, "
          f"infeasible={ego_cbf_qp['num_infeasible']}, failure={ego_cbf_qp['num_failure']}, "
          f"unbounded={ego_cbf_qp['num_unbounded']}, inaccurate={ego_cbf_qp['num_inaccurate']} "
          f"({'healthy' if ego_cbf_qp['healthy'] else 'UNHEALTHY'})")
    results = {'min_h_f': min_hf, 'min_h_r': min_hr,
               'rear_end': bool(min_hr <= 0), 'front_collision': bool(min_hf <= 0),
               'ego_cbf_qp': ego_cbf_qp}
    with open(os.path.join(run.path, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    rp = cfg['rear_params']
    footnote = (f"three_car: ego={cfg['ego_model']}(a_e={args.ego_a_decel},"
                f"gamma={args.gamma}) rear={cfg['rear_model']}(alpha=beta={rp['alpha']},"
                f"kappa={rp['kappa']},a_e={rp['a_e']},hst={rp['h_st']}) gap_r0={cfg['gap_r0']}")
    make_figure(out, t_state, t_ctrl, run.path, footnote)
    reg.commit(run, columns={'scenario': 'three_car', 'ego_model': cfg['ego_model'],
                             'rear_model': cfg['rear_model'], 'rear_kappa': rp['kappa'],
                             'rear_a_decel': rp['a_e'], 'gap_r0': cfg['gap_r0'],
                             'note': args.note})
    print(f"Registered {run.name}")


if __name__ == '__main__':
    main()
