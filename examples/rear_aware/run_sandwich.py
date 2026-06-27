"""
Sandwiched three-car demo: lead -> ego (front+rear safety) -> rear (OVM, no filter).

The *solution* counterpart to run_three_car.py (which is the forward-only motivator that gets
rear-ended). Here the ego runs the combined front+rear filter (sandwiched_cbf.SandwichedHOCBF):
a forward CBF keeps it safe behind a mildly-braking lead, while a coupled rear HOCBF (using the
ego's *assumed* rear model) keeps it from braking harder than the real follower can survive.

Because the ego cannot escape forward (the lead blocks it), the **accuracy of the assumed rear
model decides safety**:

  accurate assumed rear  -> honest rear floor clips the aggressive nominal brake to a gentle one
                            -> safe behind the lead AND no rear-end (a non-conflicting window).
  over-estimated rear    -> permissive floor lets the hard brake through -> the real sluggish
                            rear is rear-ended (min h_r < 0), though the ego stays forward-safe.

The ego nominal is deliberately *aggressive* (tailgating car-following: brakes late and hard).
One method per invocation; saves series_<method>.npz for re-plotting with plot_runs.py.

Usage:
    uv run python examples/rear_aware/run_sandwich.py                       # accurate assumed rear
    uv run python examples/rear_aware/run_sandwich.py --assumed-rear-alpha 0.8 --assumed-rear-beta 0.6
"""

import os
import json
import argparse

import numpy as np

from rear_aware_common import (plt, LW, BODY_LENGTH, L_COMBINED, REGISTER_COLUMNS,
                               resolve, registry_run, save_figure, save_run_series,
                               qp_solver_stats, _HERE)
from double_integrator_1d import DoubleIntegrator1D                # noqa: E402
from car_following_models import nominal_accel                     # noqa: E402
from rear_aware_models import rear_accel                           # noqa: E402
from run_three_car import build_lead_trace                         # noqa: E402
from sandwiched_cbf import SandwichedHOCBF                         # noqa: E402
from sandwiched_bcbf import SandwichedBackupCBF1D                  # noqa: E402
from run_registry import RunRegistry                               # noqa: E402

METHODS = ['sandwiched_hocbf', 'sandwiched_bcbf']
METHOD_STYLE = {'sandwiched_hocbf': ('sandwiched HOCBF', 'tab:blue'),
                'sandwiched_bcbf': ('sandwiched backup CBF', 'tab:green')}


def simulate_sandwich(cfg, dt, n_sim, method):
    """Lead (scripted, mild brake) -> ego (front+rear filter) -> rear (OVM, no filter)."""
    if method == 'sandwiched_hocbf':
        return _simulate_hocbf(cfg, dt, n_sim)
    if method == 'sandwiched_bcbf':
        return _simulate_bcbf(cfg, dt, n_sim)
    raise NotImplementedError(f"unknown method '{method}'")


def _simulate_hocbf(cfg, dt, n_sim):
    """Analytic combined forward-CBF ceiling + coupled rear-HOCBF floor (Stage A)."""
    L = L_COMBINED
    s_lead, v_lead = build_lead_trace(cfg['s_lead0'], n_sim, dt, cfg['lead'])

    p_safe = cfg['p_safe']
    spec = {'model': 'DoubleIntegrator1D', 'u_max': cfg['u_acc'], 'a_max': cfg['u_acc'],
            'v_max': p_safe['vbar'], 'body_length': BODY_LENGTH}
    ego = DoubleIntegrator1D(dt, dict(spec))
    rear = DoubleIntegrator1D(dt, dict(spec))

    ctrl = SandwichedHOCBF(
        forward_spec=spec, p_safe=p_safe,
        rear_model=cfg['rear_assumed_model'], rear_params=cfg['rear_assumed'],
        gamma=cfg['gamma'], L=L, d_min=cfg['d_min'],
        a_e=cfg['u_acc'], a_acc=cfg['u_acc'],
        alpha1=cfg['hocbf_a1'], alpha2=cfg['hocbf_a2'],
        robust_factor=cfg.get('hocbf_robust_factor', 1.0),
        couple_a_e_eff=cfg.get('couple_a_e_eff', True),
        a_e_eff_min=cfg.get('a_e_eff_min', 0.5))

    xe = np.array([s_lead[0] - L - cfg['gap_f0'], v_lead[0]])
    xr = np.array([xe[0] - L - cfg['gap_r0'], v_lead[0]])

    out = {k: np.zeros(n_sim + 1) for k in ('s_ego', 'v_ego', 's_rear', 'v_rear')}
    for k in ('h_f', 'h_r', 'u_ego', 'u_nom', 'u_rear', 'rhs_f', 'lb_r', 'a_e_eff'):
        out[k] = np.zeros(n_sim)
    out['conflict'] = np.zeros(n_sim)
    out['qp_status'] = []
    out['s_ego'][0], out['v_ego'][0] = xe
    out['s_rear'][0], out['v_rear'][0] = xr

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Solution may be inaccurate.*')
        for k in range(n_sim):
            lead = {'x': s_lead[k], 'vx': v_lead[k], 'length': BODY_LENGTH}
            ego_d = {'x': xe[0], 'vx': xe[1], 'length': BODY_LENGTH}
            rear_d = {'x': xr[0], 'vx': xr[1], 'length': BODY_LENGTH}

            u_nom = nominal_accel(cfg['ego_model'], ego_d, lead, cfg['ego_nom'])
            h_r = (xe[0] - L) - xr[0]
            u_e = ctrl.filter(u_nom, xe, s_lead[k], v_lead[k], h_r, xr[1])
            out['qp_status'].append(ctrl.last_status)
            u_e = max(u_e, -xe[1] / dt)

            u_r = rear_accel(rear_d, ego_d, cfg['rear_model'], cfg['rear_actual'])
            u_r = max(u_r, -xr[1] / dt)

            out['h_f'][k] = (s_lead[k] - L) - xe[0]
            out['h_r'][k] = h_r
            out['u_ego'][k] = u_e
            out['u_nom'][k] = u_nom
            out['u_rear'][k] = u_r
            out['rhs_f'][k] = ctrl.last_rhs_f
            out['lb_r'][k] = ctrl.last_lb_r
            out['a_e_eff'][k] = ctrl.last_a_e_eff
            out['conflict'][k] = float(ctrl.last_conflict)

            xe = np.array(ego.step(xe, np.array([[u_e]]))).flatten()
            xe[1] = max(xe[1], 0.0)
            xr = np.array(rear.step(xr, np.array([[u_r]]))).flatten()
            xr[1] = max(xr[1], 0.0)
            out['s_ego'][k + 1], out['v_ego'][k + 1] = xe
            out['s_rear'][k + 1], out['v_rear'][k + 1] = xr

    out['s_lead'], out['v_lead'] = s_lead, v_lead
    return out


def _simulate_bcbf(cfg, dt, n_sim):
    """Rear-aware backup CBF (Stage B): backup = follow-lead OVM + beta_f*(v_rear - v_ego)."""
    L = L_COMBINED
    s_lead, v_lead = build_lead_trace(cfg['s_lead0'], n_sim, dt, cfg['lead'])

    p_safe = cfg['p_safe']
    spec = {'model': 'DoubleIntegrator1D', 'u_max': cfg['u_acc'], 'a_max': cfg['u_acc'],
            'v_max': p_safe['vbar'], 'body_length': BODY_LENGTH}
    ego = DoubleIntegrator1D(dt, dict(spec))
    rear = DoubleIntegrator1D(dt, dict(spec))

    ctrl = SandwichedBackupCBF1D(
        robot_spec=spec, p_safe=p_safe, dt=dt, backup_horizon=cfg['backup_horizon'],
        a_e=cfg['u_acc'], u_acc=cfg['u_acc'],
        L=L, d_min=cfg['d_min'], d_f=cfg['d_f'], beta_f=cfg['backup_beta_f'],
        backup_ovm_params=cfg['backup_ovm'],
        rear_model=cfg['rear_assumed_model'], rear_params=cfg['rear_assumed'],
        rear_buffer=cfg.get('rear_buffer', 0.0),
        gamma=cfg['gamma'], gamma_terminal=cfg.get('gamma_terminal', 2.0),
        use_terminal=cfg.get('backup_terminal', False))
    ctrl.rho_rear = cfg.get('rho_rear', ctrl.rho_rear)   # ISSf needs the rear promoted to bite

    xe = np.array([s_lead[0] - L - cfg['gap_f0'], v_lead[0]])
    xr = np.array([xe[0] - L - cfg['gap_r0'], v_lead[0]])

    out = {k: np.zeros(n_sim + 1) for k in ('s_ego', 'v_ego', 's_rear', 'v_rear')}
    for k in ('h_f', 'h_r', 'u_ego', 'u_nom', 'u_rear', 'h_min', 'backup_active',
              'fwd_slack', 'rear_slack'):
        out[k] = np.zeros(n_sim)
    out['qp_status'] = []
    out['s_ego'][0], out['v_ego'][0] = xe
    out['s_rear'][0], out['v_rear'][0] = xr

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Solution may be inaccurate.*')
        for k in range(n_sim):
            lead = {'x': s_lead[k], 'vx': v_lead[k], 'length': BODY_LENGTH}
            ego_d = {'x': xe[0], 'vx': xe[1], 'length': BODY_LENGTH}
            rear_d = {'x': xr[0], 'vx': xr[1], 'length': BODY_LENGTH}

            u_nom = nominal_accel(cfg['ego_model'], ego_d, lead, cfg['ego_nom'])
            ctrl.set_nominal_controller(
                lambda X, ld=lead: np.array([[nominal_accel(
                    cfg['ego_model'],
                    {'x': float(X.flatten()[0]), 'vx': float(X.flatten()[1]),
                     'length': BODY_LENGTH}, ld, cfg['ego_nom'])]]))
            ctrl.set_lead_state(s_lead[k], v_lead[k])
            ctrl.set_rear_state(xr[0], xr[1])
            u_e = float(ctrl.solve_control_problem(np.array([xe[0], xe[1]])).flat[0])
            out['qp_status'].append(ctrl.last_status)
            u_e = max(u_e, -xe[1] / dt)

            u_r = rear_accel(rear_d, ego_d, cfg['rear_model'], cfg['rear_actual'])
            u_r = max(u_r, -xr[1] / dt)

            out['h_f'][k] = (s_lead[k] - L) - xe[0]
            out['h_r'][k] = (xe[0] - L) - xr[0]
            out['u_ego'][k] = u_e
            out['u_nom'][k] = u_nom
            out['u_rear'][k] = u_r
            out['h_min'][k] = ctrl._last_h_min
            out['backup_active'][k] = float(ctrl._using_backup)
            out['fwd_slack'][k] = ctrl.last_fwd_slack
            out['rear_slack'][k] = ctrl.last_rear_slack

            xe = np.array(ego.step(xe, np.array([[u_e]]))).flatten()
            xe[1] = max(xe[1], 0.0)
            xr = np.array(rear.step(xr, np.array([[u_r]]))).flatten()
            xr[1] = max(xr[1], 0.0)
            out['s_ego'][k + 1], out['v_ego'][k + 1] = xe
            out['s_rear'][k + 1], out['v_rear'][k + 1] = xr

    out['s_lead'], out['v_lead'] = s_lead, v_lead
    return out


def constraint_satisfaction_stats(fwd_slack, rear_slack, tol=1e-3):
    """Soft-constraint relaxation tally for the sandwiched backup CBF.

    Each per-step slack is the worst soft-constraint relaxation in its priority bucket
    (forward = priority / RHO 1e5, rear = secondary / RHO 1e3). The slack sits on the CBF
    *rate* condition (h_dot + alpha h >= 0) along the backup rollout -- NOT on h itself, so a
    slack > tol relaxes the guaranteed decay rate; it is not a collision (the barrier values
    min h_f / min h_r are reported separately). nan marks a QP-fallback (hard-brake) step.
    Forward-priority is honored iff the high forward penalty keeps the forward slack
    magnitude no larger than the rear's, so the rear bucket absorbs the sandwiched conflict.
    """
    fwd = np.asarray(fwd_slack, dtype=float)
    rear = np.asarray(rear_slack, dtype=float)
    fb = np.isnan(fwd) | np.isnan(rear)
    fwd_v = (fwd > tol) & ~fb
    rear_v = (rear > tol) & ~fb

    def _stats(arr, mask):
        clean = arr[~fb]
        return {'relaxed_steps': int(mask.sum()),
                'max_slack': float(np.max(clean)) if clean.size else 0.0,
                'mean_slack_when_relaxed': float(arr[mask].mean()) if mask.any() else 0.0}

    fp, rs = _stats(fwd, fwd_v), _stats(rear, rear_v)
    return {'total_steps': int(fwd.size), 'tol': tol, 'fallback_steps': int(fb.sum()),
            'forward_priority': fp, 'rear_secondary': rs,
            'fwd_to_rear_slack_ratio': (fp['max_slack'] / rs['max_slack']
                                        if rs['max_slack'] > 0 else 0.0),
            'priority_respected': bool(fp['max_slack'] <= rs['max_slack'] + tol)}


def make_figure(method, out, cfg, t_state, t_ctrl, save_dir, footnote):
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    axp, axg, axv, axu = axes
    tag, c = METHOD_STYLE[method]

    axp.plot(t_state, out['s_lead'], 'k-', lw=LW, label='Lead')
    axp.plot(t_state, out['s_ego'], color=c, lw=LW, label=f'Ego ({tag})')
    axp.plot(t_state, out['s_rear'], color='tab:orange', lw=LW, label='Rear (OVM, no filter)')
    axp.set_ylabel('position [m]'); axp.set_title('Positions'); axp.legend(); axp.grid(alpha=0.3)

    axg.plot(t_ctrl, out['h_f'], color='tab:blue', lw=LW, label='h_f (ego - lead)')
    axg.plot(t_ctrl, out['h_r'], color='tab:orange', lw=LW, label='h_r (rear - ego)')
    axg.axhline(0.0, color='red', ls=':', lw=2, label='collision (h=0)')
    axg.axhline(cfg['d_min'], color='gray', ls='--', lw=1.5, label=f"d_min={cfg['d_min']}")
    axg.set_ylabel('gap [m]'); axg.set_title('Forward gap h_f vs rear gap h_r')
    axg.legend(ncol=2); axg.grid(alpha=0.3)

    axv.plot(t_state, out['v_lead'], 'k-', lw=LW, label='Lead')
    axv.plot(t_state, out['v_ego'], color=c, lw=LW, label='Ego')
    axv.plot(t_state, out['v_rear'], color='tab:orange', lw=LW, label='Rear')
    axv.set_ylabel('velocity [m/s]'); axv.set_title('Velocities'); axv.legend(); axv.grid(alpha=0.3)

    axu.plot(t_ctrl, out['u_nom'], color='gray', lw=LW * 0.8, ls=':', label='ego nominal')
    axu.plot(t_ctrl, out['u_ego'], color=c, lw=LW, label=f'ego ({tag})')
    if method == 'sandwiched_hocbf':
        axu.plot(t_ctrl, out['rhs_f'], color='tab:blue', lw=1.2, ls='--', label='forward ceiling rhs_f')
        axu.plot(t_ctrl, out['lb_r'], color='tab:orange', lw=1.2, ls='--', label='rear floor lb_r')
        axu.plot(t_ctrl, -out['a_e_eff'], color='tab:red', lw=1.2, ls='-.', label='-a_e_eff (rear-aware brake)')
    else:
        axu.plot(t_ctrl, out['h_min'], color='tab:red', lw=1.2, ls='-.', label='backup min-h (fwd&rear)')
    axu.set_ylabel('ego accel [m/s^2]'); axu.set_xlabel('time [s]'); axu.set_title('Ego control + bounds')
    axu.legend(ncol=2, fontsize=9); axu.grid(alpha=0.3)

    fig.suptitle(f'Sandwiched three-car ({tag})')
    save_figure(fig, save_dir, 'sandwich.png', footnote)


def build_cfg(args, method=None):
    if method is None:
        method = getattr(args, 'method', 'sandwiched_hocbf')
    lead_params = {'v_cruise': args.lead_v_cruise, 't_cruise': args.lead_t_cruise,
                   'a_brake': args.lead_a_brake, 't_hold_dur': args.lead_t_hold,
                   'a_accel': 2.0}
    # Aggressive ego nominal: tailgating car-following (small standstill gap, strong
    # reaction) -> brakes late and hard when the lead slows.
    ego_nom = {'alpha': args.ego_alpha, 'beta': args.ego_beta, 'kappa': args.ego_kappa,
               'h_st': args.ego_hst, 'v0': args.lead_v_cruise, 'v_max': args.lead_v_cruise,
               'a_max': args.u_acc, 'a_e': args.u_acc, 'T': 0.4, 's0': 1.0, 'b_comfort': 2.0}
    rear_actual = {'alpha': resolve(args.rear_alpha, 0.40),
                   'beta': resolve(args.rear_beta, 0.32),
                   'kappa': resolve(args.rear_kappa, 0.7),
                   'h_st': resolve(args.rear_hst, 2.0),
                   'v0': args.lead_v_cruise, 'v_max': args.lead_v_cruise,
                   'a_max': resolve(args.rear_a_max, 2.0),
                   'a_e': resolve(args.rear_a_decel, 2.5),
                   'T': resolve(args.rear_T, 1.0), 's0': 2.0,
                   'b_comfort': resolve(args.rear_b, 2.0)}
    rear_assumed = dict(rear_actual)
    for key, val in [('alpha', args.assumed_rear_alpha), ('beta', args.assumed_rear_beta),
                     ('kappa', args.assumed_rear_kappa), ('a_e', args.assumed_rear_a_decel)]:
        if val is not None:
            rear_assumed[key] = val
    cfg = {
        'scenario': 'sandwich', 'method': method, 'ego_model': args.ego_model,
        's_lead0': 60.0, 'gap_f0': args.gap_f0, 'gap_r0': resolve(args.gap_r0, 6.0),
        'd_min': args.d_min, 'u_acc': args.u_acc, 'gamma': args.gamma,
        'p_safe': {'tau': args.tau, 'a_e': args.u_acc, 'a_l': args.a_lead_decel,
                   'vbar': args.lead_v_cruise + 3.0},
        'lead': lead_params, 'ego_nom': ego_nom,
        'rear_model': args.rear_model, 'rear_actual': rear_actual,
        'rear_assumed_model': resolve(args.assumed_rear_model, args.rear_model),
        'rear_assumed': rear_assumed,
    }
    if method == 'sandwiched_hocbf':
        cfg.update({'hocbf_a1': args.hocbf_a1, 'hocbf_a2': args.hocbf_a2,
                    'hocbf_robust_factor': args.hocbf_robust_factor,
                    'couple_a_e_eff': not args.no_couple_a_e_eff,
                    'a_e_eff_min': args.a_e_eff_min})
    elif method == 'sandwiched_bcbf':
        cfg.update({'d_f': args.d_f, 'backup_horizon': args.backup_horizon,
                    'backup_beta_f': args.backup_beta_f,
                    'gamma_terminal': args.gamma_terminal,
                    'backup_terminal': not args.no_backup_terminal,
                    'backup_ovm': {'alpha': args.backup_alpha, 'beta': args.backup_beta,
                                   'kappa': args.backup_kappa, 'h_st': args.backup_hst,
                                   'v_max': args.lead_v_cruise + 3.0}})
        # ISSf rear margin: inflate d_min by L_inter*|dparam| + residue, |dparam| the
        # assumed-vs-actual alpha+beta mismatch (the disturbance the assumed model carries).
        # Only recorded when active, so margin-off configs/fingerprints match the pre-ISSf runs.
        issf_active = (args.l_inter_ratio != 0.0 or args.l_inter_residue != 0.0
                       or args.rho_rear != 1e3)
        if issf_active:
            param_mismatch = (abs(rear_assumed['alpha'] - rear_actual['alpha'])
                              + abs(rear_assumed['beta'] - rear_actual['beta']))
            cfg.update({'l_inter_ratio': args.l_inter_ratio,
                        'l_inter_residue': args.l_inter_residue,
                        'param_mismatch': param_mismatch,
                        'rear_buffer': args.l_inter_ratio * param_mismatch + args.l_inter_residue,
                        'rho_rear': args.rho_rear})
    return cfg


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', default='sandwiched_hocbf', choices=METHODS)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tf', type=float, default=20.0)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--ego-model', choices=['ovm', 'idm'], default='ovm')
    p.add_argument('--u-acc', type=float, default=3.0, help='ego accel/brake magnitude')
    p.add_argument('--a-lead-decel', type=float, default=1.5,
                   help='lead max decel assumed by the forward CBF (a_l); matched to lead brake')
    p.add_argument('--gap-f0', type=float, default=14.0, help='initial ego-lead gap')
    p.add_argument('--gap-r0', type=float, default=5.0, help='initial rear-ego gap')
    p.add_argument('--d-min', type=float, default=1.0, help='rear collision margin')
    # Lead (scripted, mild brake by default to leave a non-conflicting window).
    p.add_argument('--lead-v-cruise', type=float, default=10.0)
    p.add_argument('--lead-t-cruise', type=float, default=3.0)
    p.add_argument('--lead-a-brake', type=float, default=1.5, help='lead braking decel (mild)')
    p.add_argument('--lead-t-hold', type=float, default=4.0)
    # Aggressive ego nominal (tailgating).
    p.add_argument('--ego-alpha', type=float, default=0.8)
    p.add_argument('--ego-beta', type=float, default=0.5)
    p.add_argument('--ego-kappa', type=float, default=1.0)
    p.add_argument('--ego-hst', type=float, default=1.0)
    # HOCBF gains.
    p.add_argument('--hocbf-a1', type=float, default=1.0)
    p.add_argument('--hocbf-a2', type=float, default=1.0)
    p.add_argument('--hocbf-robust-factor', type=float, default=1.0)
    # Backup CBF (Stage B): rear-aware OVM backup + forward/rear barriers along the rollout.
    p.add_argument('--backup-horizon', type=float, default=6.0)
    p.add_argument('--backup-beta-f', type=float, default=1.0,
                   help='rear-aware softening gain beta_f in the backup OVM (the key knob)')
    p.add_argument('--backup-alpha', type=float, default=0.6, help='backup OVM follow-lead alpha')
    p.add_argument('--backup-beta', type=float, default=0.5, help='backup OVM follow-lead beta')
    p.add_argument('--backup-kappa', type=float, default=0.5)
    p.add_argument('--backup-hst', type=float, default=5.0)
    p.add_argument('--d-f', type=float, default=0.5, help='forward collision margin (backup CBF)')
    # ISSf error margin on the rear barrier: rear_buffer = l_inter_ratio*|dparam| + l_inter_residue,
    # where |dparam| = |alpha_assume-alpha_actual| + |beta_assume-beta_actual| (opt-in; 0 = off).
    p.add_argument('--l-inter-ratio', type=float, default=0.0,
                   help='ISSf rear-margin gain L_inter (m per unit alpha+beta mismatch); 0 = off')
    p.add_argument('--l-inter-residue', type=float, default=0.0,
                   help='ISSf rear-margin constant residue L_inter_residue [m]; 0 = off')
    p.add_argument('--rho-rear', type=float, default=1e3,
                   help='rear soft-constraint penalty (authority). Default 1e3 keeps the rear '
                        'secondary (forward 1e5); the ISSf margin only bites once the rear '
                        'constraint is promoted (e.g. 3e4) so it can shape the brake.')
    p.add_argument('--gamma-terminal', type=float, default=2.0)
    p.add_argument('--no-backup-terminal', action='store_true',
                   help='drop the terminal gap constraints (default: terminals off)',
                   default=True)
    p.add_argument('--backup-terminal', dest='no_backup_terminal', action='store_false',
                   help='enable the terminal gap constraints')
    # Rear-aware forward CBF (a_e_eff coupling).
    p.add_argument('--no-couple-a-e-eff', action='store_true',
                   help='disable a_e_eff coupling (naive fixed-a_e front-ceiling/rear-floor clip)')
    p.add_argument('--a-e-eff-min', type=float, default=0.5,
                   help='floor on the rear-aware braking authority a_e_eff')
    # Rear follower (actual) + what the filter assumes.
    p.add_argument('--rear-model', choices=['ovm', 'idm'], default='ovm')
    p.add_argument('--assumed-rear-model', choices=['ovm', 'idm'], default=None)
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
    p.add_argument('--output-dir', default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--override', action='store_true')
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

    out = simulate_sandwich(cfg, dt, n_sim, method)
    save_run_series(run.path, method, t_state, t_ctrl, out)

    min_hf, min_hr = float(out['h_f'].min()), float(out['h_r'].min())
    mismatch = bool(cfg['rear_assumed'] != cfg['rear_actual']
                    or cfg['rear_assumed_model'] != cfg['rear_model'])
    print(f"  min h_f = {min_hf:7.3f} m ({'SAFE' if min_hf > 0 else 'FRONT-COLLIDE'} vs lead)")
    print(f"  min h_r = {min_hr:7.3f} m ({'rear-end!' if min_hr <= 0 else 'safe'} from rear)")

    status_stats = qp_solver_stats(out['qp_status'])
    results = {'method': method, 'param_mismatch': mismatch,
               'min_h_f': min_hf, 'min_h_r': min_hr,
               'front_collision': bool(min_hf <= 0), 'rear_end': bool(min_hr <= 0),
               'status_stats': status_stats}
    if 'conflict' in out:                       # HOCBF-only diagnostic
        conflict_frac = float(np.mean(out['conflict']))
        results['conflict_fraction'] = conflict_frac
        print(f"  conflict fraction = {conflict_frac:.3f}  (rear floor > forward ceiling)")
    else:                                       # bcbf: backup usage + constraint satisfaction
        backup_frac = float(np.mean(out['backup_active']))
        results['backup_fraction'] = backup_frac
        # ISSf margin diagnostics + control-effort cost (the narrative's performance metric).
        effort = float(np.sum(np.abs(out['u_ego'] - out['u_nom'])) * dt)
        results.update({'param_mismatch_mag': float(cfg.get('param_mismatch', 0.0)),
                        'rear_buffer': float(cfg.get('rear_buffer', 0.0)),
                        'l_inter_ratio': float(cfg.get('l_inter_ratio', 0.0)),
                        'l_inter_residue': float(cfg.get('l_inter_residue', 0.0)),
                        'rear_clears_dmin': bool(min_hr > cfg['d_min']),
                        'control_effort_integral': effort})
        csat = constraint_satisfaction_stats(out['fwd_slack'], out['rear_slack'])
        results['constraint_satisfaction'] = csat
        fp, rs = csat['forward_priority'], csat['rear_secondary']
        usable = status_stats['total_steps'] - status_stats['num_infeasible'] - status_stats['num_failure']
        print(f"  rear ISSf buffer = {results['rear_buffer']:.3f} m  "
              f"(mismatch |dα|+|dβ|={results['param_mismatch_mag']:.3f}, "
              f"L_inter={results['l_inter_ratio']:g}, residue={results['l_inter_residue']:g})")
        print(f"  rear clears d_min ({cfg['d_min']}): {results['rear_clears_dmin']}  "
              f"|  control effort integral(|u-u_nom| dt) = {effort:.3f}")
        print(f"  backup-active fraction = {backup_frac:.3f}")
        print(f"  QP status: optimal/usable={usable}  infeasible={status_stats['num_infeasible']}"
              f"  failure={status_stats['num_failure']}  fallback(hard-brake)={csat['fallback_steps']}")
        print(f"  CBF-rate relaxations (slack > {csat['tol']:g}) over {csat['total_steps']} steps:")
        print(f"    forward (priority)  : {fp['relaxed_steps']:3d} steps  max slack {fp['max_slack']:.2e}")
        print(f"    rear    (secondary) : {rs['relaxed_steps']:3d} steps  max slack {rs['max_slack']:.2e}")
        print(f"    fwd/rear max-slack ratio = {csat['fwd_to_rear_slack_ratio']:.2f}  "
              f"-> forward priority {'respected' if csat['priority_respected'] else 'VIOLATED'}"
              f" (forward kept tighter)")
    with open(os.path.join(run.path, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    rp, ra = cfg['rear_actual'], cfg['rear_assumed']
    footnote = (f"sandwich[{method}]: lead(a_brake={cfg['lead']['a_brake']}) "
                f"ego={cfg['ego_model']}(aggr) rear={cfg['rear_model']}"
                f"(a/b={rp['alpha']}/{rp['beta']}) assumed(a/b={ra['alpha']}/{ra['beta']}) "
                f"gap_f0={cfg['gap_f0']} gap_r0={cfg['gap_r0']} mismatch={mismatch}")
    make_figure(method, out, cfg, t_state, t_ctrl, run.path, footnote)
    reg.commit(run, columns={'scenario': 'sandwich', 'method': method,
                             'ego_model': cfg['ego_model'],
                             'rear_model': cfg['rear_model'],
                             'assumed_rear_model': cfg['rear_assumed_model'],
                             'rear_alpha': rp['alpha'], 'rear_beta': rp['beta'],
                             'assumed_rear_alpha': ra['alpha'], 'assumed_rear_beta': ra['beta'],
                             'rear_kappa': rp['kappa'], 'rear_a_decel': rp['a_e'],
                             'gap_r0': cfg['gap_r0'], 'mismatch': mismatch, 'note': args.note})
    print(f"Registered {run.name}")


if __name__ == '__main__':
    main()
