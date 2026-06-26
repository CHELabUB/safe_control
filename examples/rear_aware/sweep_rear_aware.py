"""
Parameter sweep for the rear-aware scenarios.

Runs a grid of configurations (fast: metrics only, no figures), writes a summary
CSV, and prints the representative cases:

  three_car : configurations where the ego stays safe vs the lead but is rear-ended
              (min h_r < 0)  -> motivating cases (rendered by run_three_car.py).
  ego_rear  : configurations where the baseline is rear-ended but the coupled HOCBF
              keeps it safe -> cases rendered by run_ego_rear.py.

The full figure + registry entry for a chosen case is then produced by
run_three_car.py / run_ego_rear.py with the matching flags.

Usage:
    uv run python examples/rear_aware/sweep_rear_aware.py --scenario three_car
    uv run python examples/rear_aware/sweep_rear_aware.py --scenario ego_rear
"""

import os
import sys
import csv
import argparse
import itertools
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np                                   # noqa: E402
import run_three_car as three_car                    # noqa: E402
import run_ego_rear as ego_rear                       # noqa: E402

warnings.filterwarnings('ignore')


def _args(parser, **overrides):
    a = parser.parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def sweep_three_car(out_csv):
    dt = 0.05
    n = int(round(14.0 / dt))
    tb = int(round(3.0 / dt))                          # lead starts braking at t=3
    rows = []
    # Realistic framing: a steady (near-equilibrium) follower that rear-ends by
    # *slow reaction*. Scan responsiveness (alpha=beta), headway slope kappa
    # (-> equilibrium gap), and the near-equilibrium initial gap.
    grid = itertools.product([0.15, 0.2, 0.25],        # rear alpha = beta
                             [0.6, 0.7, 0.8],           # rear_kappa
                             [16.0, 18.0, 20.0])        # gap_r0
    for ab, rk, gap in grid:
        a = _args(three_car.build_parser(), rear_alpha=ab, rear_beta=ab,
                  rear_kappa=rk, rear_hst=2.0, rear_a_decel=2.5, gap_r0=gap)
        cfg = three_car.build_cfg(a)
        out = three_car.simulate_three_car(cfg, dt, n)
        min_hf, min_hr = float(out['h_f'].min()), float(out['h_r'].min())
        hr = out['h_r']
        drift = float(hr[:tb].max() - hr[:tb].min())   # ~0 => steady before braking
        rows.append({'rear_ab': ab, 'rear_kappa': rk, 'gap_r0': gap,
                     'eq_gap': round(10.0 / rk + 2.0, 1),
                     'prebrake_drift': round(drift, 3),
                     'min_h_f': round(min_hf, 3), 'min_h_r': round(min_hr, 3),
                     'steady_start': drift < 0.1,
                     'safe_vs_lead': min_hf > 0, 'rear_end': min_hr <= 0})
    _write(out_csv, rows)
    good = [r for r in rows if r['safe_vs_lead'] and r['rear_end'] and r['steady_start']]
    print(f"\n{len(good)}/{len(rows)} configs: steady start, ego safe vs lead, AND "
          f"rear-ended. Representative (mild crash):")
    for r in sorted(good, key=lambda r: abs(r['min_h_r'] + 3))[:5]:
        print(f"  alpha=beta={r['rear_ab']} kappa={r['rear_kappa']} gap_r0={r['gap_r0']} "
              f"(eq={r['eq_gap']})  ->  min_h_r={r['min_h_r']}")


def sweep_ego_rear(out_csv):
    dt = 0.05
    n = int(round(9.0 / dt))
    rows = []
    grid = itertools.product([8.0, 9.0, 10.0],       # ego_v0
                             [16.0, 18.0, 20.0],      # stop_distance
                             [2.0, 3.0, 4.0])         # gap_r0
    for v0, stop, gap in grid:
        a = _args(ego_rear.build_parser(), ego_v0=v0, stop_distance=stop, gap_r0=gap)
        cfg = ego_rear.build_cfg(a)
        b = float(ego_rear.simulate_ego_rear(cfg, dt, n, 'baseline')['h_r'].min())
        h = float(ego_rear.simulate_ego_rear(cfg, dt, n, 'hocbf')['h_r'].min())
        rows.append({'ego_v0': v0, 'stop_distance': stop, 'gap_r0': gap,
                     'baseline_min_h_r': round(b, 3), 'hocbf_min_h_r': round(h, 3),
                     'baseline_rear_end': b <= 0, 'hocbf_safe': h > 0,
                     'filter_saves': (b <= 0 and h > 0)})
    _write(out_csv, rows)
    good = [r for r in rows if r['filter_saves']]
    print(f"\n{len(good)}/{len(rows)} configs: baseline rear-ends BUT the HOCBF "
          f"keeps it safe. Representative:")
    for r in sorted(good, key=lambda r: r['baseline_min_h_r'])[:5]:
        print(f"  ego_v0={r['ego_v0']} stop_distance={r['stop_distance']} "
              f"gap_r0={r['gap_r0']}  ->  baseline={r['baseline_min_h_r']} "
              f"hocbf={r['hocbf_min_h_r']}")


def _settle_time(v_ego, dt, v_desired, band):
    """Time (s) after which |v_ego - v_desired| stays <= band; (tf, False) if it never does.

    Scans from the end: the settle index is the first sample from which every later sample
    is within band -- i.e. one past the last out-of-band sample. A run that ends out of band
    never settles (the conservative/pinned case), reported as (tf, settled=False).
    """
    within = np.abs(np.asarray(v_ego) - v_desired) <= band
    tf = (len(v_ego) - 1) * dt
    if not within[-1]:
        return tf, False
    k = len(within) - 1
    while k > 0 and within[k - 1]:
        k -= 1
    return k * dt, True


def _sim_metrics(cfg, dt, n, mode, v_desired=None, settle_band=0.5):
    """Run one ego_rear simulation and reduce it to scalar metrics.

    Safety axis: min_h_r / rear_end. Performance axes (all vs the nominal task of reaching
    v_desired): t_settle, v_offset (steady offset above target), control_effort (integral u^2),
    and intervention (integral |u - u_nom|, how hard the filter overrides the driver).
    v_final is averaged over the last ~1 s (robust to end-of-horizon ripple).
    """
    out = ego_rear.simulate_ego_rear(cfg, dt, n, mode)
    nset = max(1, int(round(1.0 / dt)))
    hmin = float(out['h_r'].min())
    v_final = float(np.mean(out['v_ego'][-nset:]))
    u_ego, u_nom = out['u_ego'], out['u_nom']
    m = {
        'min_h_r': hmin,
        'v_final': v_final,
        'peak_v': float(out['v_ego'].max()),
        'rear_end': bool(hmin <= 0),
        'control_effort': float(np.sum(u_ego ** 2) * dt),
        'intervention': float(np.sum(np.abs(u_ego - u_nom)) * dt),
        'peak_u': float(np.max(np.abs(u_ego))),
        'qp_infeasible': int(sum(1 for s in out['qp_status'] if 'infeasible' in str(s))),
    }
    if v_desired is not None:
        t_set, settled = _settle_time(out['v_ego'], dt, v_desired, settle_band)
        m['t_settle'] = t_set
        m['settled'] = settled
        m['v_offset'] = v_final - v_desired
    return m


def sweep_v_regime(out_csv, base):
    """Phase 1: find an (ego_v0, v_desired, v_max) operating point with *room*.

    For each cell, run the baseline (no filter) and the accurate backup CBF (assumed rear
    == actual rear). A 'good_regime' cell is one where the baseline is rear-ended yet the
    accurate filter is both safe AND settles near v_desired (not pinned at v_max) -- i.e.
    the comparison is non-degenerate (baseline crashes, accurate genuinely does better on
    both safety and task). The accurate-case t_settle / control_effort are reported so an
    operating point with a pronounced conservative-vs-accurate performance gap can be picked.
    Initial speed ego_v0 is swept too (higher v0 => harder brake => more room).
    """
    dt = 0.05
    n = int(round(base['tf'] / dt))
    band = base['settle_band']
    rows = []
    for v0, vd, vmax in itertools.product(base['ego_v0_grid'], base['v_desired_grid'],
                                          base['v_max_grid']):
        a = _args(ego_rear.build_parser(),
                  ego_target='speed', v_desired=vd, v_max=vmax, v_road=base['v_road'],
                  ego_v0=v0, gap_r0=base['gap_r0'], d_min=base['d_min'],
                  u_acc=base['u_acc'], backup_horizon=base['backup_horizon'],
                  rear_alpha=base['rear_alpha'], rear_beta=base['rear_beta'])
        b = _sim_metrics(ego_rear.build_cfg(a, 'baseline'), dt, n, 'baseline', vd, band)
        f = _sim_metrics(ego_rear.build_cfg(a, 'bcbf'), dt, n, 'bcbf', vd, band)
        pinned = f['v_final'] > vmax - 0.5
        settles = f['settled'] and not pinned
        rows.append({
            'ego_v0': v0, 'v_desired': vd, 'v_max': vmax,
            'baseline_min_h_r': round(b['min_h_r'], 3),
            'baseline_v_final': round(b['v_final'], 2),
            'baseline_rear_end': b['rear_end'],
            'accurate_min_h_r': round(f['min_h_r'], 3),
            'accurate_v_final': round(f['v_final'], 2),
            'accurate_v_offset': round(f['v_offset'], 2),
            'accurate_t_settle': round(f['t_settle'], 2),
            'accurate_control_effort': round(f['control_effort'], 2),
            'accurate_peak_v': round(f['peak_v'], 2),
            'accurate_safe': f['min_h_r'] > 0,
            'accurate_settles': settles,
            'accurate_pinned': pinned,
            'good_regime': b['rear_end'] and (f['min_h_r'] > 0) and settles,
        })
    _write(out_csv, rows)
    good = [r for r in rows if r['good_regime']]
    print(f"\n{len(good)}/{len(rows)} cells: baseline rear-ends AND accurate filter is "
          f"safe AND settles near v_desired (non-degenerate). Representative:")
    for r in sorted(good, key=lambda r: r['accurate_v_offset'])[:6]:
        print(f"  ego_v0={r['ego_v0']} v_desired={r['v_desired']} v_max={r['v_max']}  ->  "
              f"baseline(min_h_r={r['baseline_min_h_r']}) "
              f"accurate(min_h_r={r['accurate_min_h_r']}, v_off={r['accurate_v_offset']}, "
              f"t_set={r['accurate_t_settle']}, effort={r['accurate_control_effort']})")
    if not good:
        print("  (none — widen v_max / lower v_desired, or raise actual rear responsiveness)")


def sweep_interaction_accuracy(out_csv, base):
    """At a fixed operating point, sweep the *assumed* rear responsiveness.

    A single scale multiplies the assumed (alpha, beta) the backup CBF believes; the actual
    rear is held fixed. The axis spans passive(0) -> under(<1) -> accurate(1) -> over(>1):

      baseline : reaches v_desired but is rear-ended       (good perf, unsafe)
      passive  : ego pinned high, never slows               (safe, saturated corner)
      under    : ego settles high (distrusts rear braking)  (safe, degraded perf)
      accurate : safe AND settles near v_desired            (best)
      over     : over-trusts rear braking                   (safe HERE -- forward escape; the
                 over->collision arm needs the sandwiched three-car scenario)

    Records the four performance metrics (t_settle, v_offset, control_effort, intervention)
    so the conservative-vs-accurate degradation is quantified, not just min_h_r / v_final.
    scale=0 is the saturated 'passive' corner (labeled, not part of the gradient).
    """
    dt = 0.05
    n = int(round(base['tf'] / dt))
    vd, vmax, band = base['v_desired'], base['v_max'], base['settle_band']
    common = dict(ego_target='speed', v_desired=vd, v_max=vmax, v_road=base['v_road'],
                  ego_v0=base['ego_v0'], gap_r0=base['gap_r0'], d_min=base['d_min'],
                  u_acc=base['u_acc'], backup_horizon=base['backup_horizon'],
                  rear_alpha=base['rear_alpha'], rear_beta=base['rear_beta'])

    def _row(scale, mt, aa, bb, m):
        return {'scale': scale, 'mismatch_type': mt,
                'assumed_alpha': aa, 'assumed_beta': bb,
                'min_h_r': round(m['min_h_r'], 3), 'rear_end': m['rear_end'],
                'v_final': round(m['v_final'], 2),
                'v_offset': round(m['v_offset'], 2),
                't_settle': round(m['t_settle'], 2), 'settled': m['settled'],
                'control_effort': round(m['control_effort'], 2),
                'intervention': round(m['intervention'], 2),
                'peak_u': round(m['peak_u'], 2), 'qp_infeasible': m['qp_infeasible']}

    rows = []
    # Baseline reference (no filter): the unsafe-but-on-task corner.
    mb = _sim_metrics(ego_rear.build_cfg(_args(ego_rear.build_parser(), **common),
                                         'baseline'), dt, n, 'baseline', vd, band)
    rows.append(_row('', 'baseline', '', '', mb))
    for scale in base['scales']:
        aa, bb = base['rear_alpha'] * scale, base['rear_beta'] * scale
        a = _args(ego_rear.build_parser(), assumed_rear_alpha=aa, assumed_rear_beta=bb,
                  **common)
        m = _sim_metrics(ego_rear.build_cfg(a, 'bcbf'), dt, n, 'bcbf', vd, band)
        mt = ('passive' if scale == 0 else 'under' if scale < 1
              else 'accurate' if scale == 1 else 'over')
        rows.append(_row(scale, mt, round(aa, 3), round(bb, 3), m))
    _write(out_csv, rows)
    print(f"\nInteraction-accuracy progression (actual alpha={base['rear_alpha']}, "
          f"beta={base['rear_beta']}; ego_v0={base['ego_v0']}, v_desired={vd}, v_max={vmax}):")
    hdr = ('type', 'scale', 'min_h_r', 'v_off', 't_set', 'effort', 'interv')
    print('  {:9s} {:>6s} {:>9s} {:>6s} {:>6s} {:>7s} {:>7s}  outcome'.format(*hdr))
    for r in rows:
        tag = 'REAR-END' if r['rear_end'] else ('safe' if r['settled'] else 'safe,no-settle')
        print('  {:9s} {:>6s} {:>9s} {:>6s} {:>6s} {:>7s} {:>7s}  {}'.format(
            r['mismatch_type'], str(r['scale']), str(r['min_h_r']), str(r['v_offset']),
            str(r['t_settle']), str(r['control_effort']), str(r['intervention']), tag))


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def _floats(s):
    """Parse a comma-separated list of floats (e.g. '0,0.5,1.0')."""
    return [float(x) for x in str(s).split(',') if x.strip() != '']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', default='three_car',
                   choices=['three_car', 'ego_rear', 'v_regime', 'interaction_accuracy'])
    p.add_argument('--output-dir', default=None,
                   help='results directory (default: <example>/temp_<scenario> for the '
                        'investigation sweeps v_regime / interaction_accuracy, else '
                        '<example>/output). temp_* folders are gitignored.')
    # Shared scenario knobs (used by v_regime / interaction_accuracy).
    p.add_argument('--tf', type=float, default=None,
                   help='sim horizon [s] (default: 25 for interaction_accuracy so the '
                        'accurate run settles; 15 for v_regime)')
    p.add_argument('--ego-v0', type=float, default=9.0,
                   help='initial ego speed (interaction_accuracy operating point)')
    p.add_argument('--gap-r0', type=float, default=3.0)
    p.add_argument('--d-min', type=float, default=1.0)
    p.add_argument('--u-acc', type=float, default=3.0)
    p.add_argument('--v-road', type=float, default=12.0)
    p.add_argument('--backup-horizon', type=float, default=5.0)
    p.add_argument('--rear-alpha', type=float, default=0.4, help='ACTUAL rear alpha')
    p.add_argument('--rear-beta', type=float, default=0.3, help='ACTUAL rear beta')
    p.add_argument('--settle-band', type=float, default=0.5,
                   help='|v_ego - v_desired| <= band counts as settled to target')
    # v_regime grids (ego_v0 swept too: higher v0 => harder brake => more room).
    p.add_argument('--ego-v0-grid', type=_floats, default='7,9,11')
    p.add_argument('--v-desired-grid', type=_floats, default='0.5,1.0,2.0')
    p.add_argument('--v-max-grid', type=_floats, default='13,14,15')
    # interaction_accuracy fixed operating point + assumed-responsiveness scale axis.
    p.add_argument('--v-desired', type=float, default=1.0)
    p.add_argument('--v-max', type=float, default=14.0)
    p.add_argument('--scales', type=_floats, default='0,0.5,0.75,1.0,1.25,1.5,1.75,2.0',
                   help='assumed/actual (alpha,beta) ratios; 0=passive, 1=accurate, >1=over')
    args = p.parse_args()

    if args.tf is None:
        args.tf = 25.0 if args.scenario == 'interaction_accuracy' else 15.0
    default_sub = ('output' if args.scenario in ('three_car', 'ego_rear')
                   else f'temp_{args.scenario}')
    out_dir = args.output_dir or os.path.join(_HERE, default_sub)
    csv_path = os.path.join(out_dir, f'sweep_{args.scenario}.csv')
    base = {k: getattr(args, k) for k in
            ('tf', 'ego_v0', 'gap_r0', 'd_min', 'u_acc', 'v_road', 'backup_horizon',
             'rear_alpha', 'rear_beta', 'settle_band', 'ego_v0_grid', 'v_desired_grid',
             'v_max_grid', 'v_desired', 'v_max', 'scales')}
    if args.scenario == 'three_car':
        sweep_three_car(csv_path)
    elif args.scenario == 'ego_rear':
        sweep_ego_rear(csv_path)
    elif args.scenario == 'v_regime':
        sweep_v_regime(csv_path, base)
    else:
        sweep_interaction_accuracy(csv_path, base)


if __name__ == '__main__':
    main()
