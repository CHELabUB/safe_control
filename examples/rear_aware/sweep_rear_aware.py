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


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=['three_car', 'ego_rear'], default='three_car')
    p.add_argument('--output-dir', default=None,
                   help='results directory (default: <example>/output)')
    args = p.parse_args()
    out_dir = args.output_dir or os.path.join(_HERE, 'output')
    csv_path = os.path.join(out_dir, f'sweep_{args.scenario}.csv')
    if args.scenario == 'three_car':
        sweep_three_car(csv_path)
    else:
        sweep_ego_rear(csv_path)


if __name__ == '__main__':
    main()
