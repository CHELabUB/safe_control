"""
Statistics plot for the interaction-accuracy sweep (sweep_rear_aware.py --scenario
interaction_accuracy).

Reads that sweep's CSV and plots the metrics against the assumed/actual rear-responsiveness
scale, so the *trend* is visible at a glance: as the backup CBF's assumed rear model goes
passive(0) -> under(<1) -> accurate(1) -> over(>1),

  - safety  (min_h_r) stays flat & safe          -> the structural robustness finding
  - task    (t_settle, v_offset) improve         -> conservative assumptions perform poorly
  - effort / intervention                         -> how hard the filter fights the driver

The baseline (no filter) row is drawn as a horizontal reference (it rear-ends, so its safety
point is also marked). This is the companion "statistics" figure to the per-run *behavior*
overlay produced by plot_runs.py.

Usage:
    uv run python examples/rear_aware/plot_accuracy_sweep.py \
        examples/rear_aware/temp_interaction_accuracy/sweep_interaction_accuracy.csv
"""

import os
import csv
import argparse

import numpy as np

from rear_aware_common import plt, save_figure        # noqa: E402

# color per mismatch type (matches the narrative buckets)
TYPE_COLOR = {'baseline': 'tab:red', 'passive': 'tab:purple', 'under': 'tab:orange',
              'accurate': 'tab:green', 'over': 'tab:blue'}


def _read(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f'empty sweep CSV: {path}')
    return rows


def _f(row, key):
    """Float or None for a possibly-blank cell."""
    v = row.get(key, '')
    return None if v in ('', None) else float(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', help='path to sweep_interaction_accuracy.csv')
    ap.add_argument('--output', default=None,
                    help='output figure path (default: <csv_dir>/accuracy_sweep.png)')
    args = ap.parse_args()

    rows = _read(args.csv)
    base = [r for r in rows if r['mismatch_type'] != 'baseline']
    bl = next((r for r in rows if r['mismatch_type'] == 'baseline'), None)
    scale = np.array([float(r['scale']) for r in base])
    colors = [TYPE_COLOR.get(r['mismatch_type'], 'gray') for r in base]

    # (key, axis title, y-label, baseline-as-reference?)
    panels = [('min_h_r', 'Safety: min rear gap', 'min h_r [m]', True),
              ('t_settle', 'Task: time to settle to v_desired', 't_settle [s]', True),
              ('v_offset', 'Task: steady speed offset (v_final - v_desired)', 'v_offset [m/s]', True),
              ('intervention', 'Filter intervention from nominal', 'int |u-u_nom| dt', False),
              ('control_effort', 'Control effort', 'int u^2 dt', True),
              ('peak_u', 'Peak |accel|', 'max |u| [m/s^2]', True)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title, ylab, show_bl) in zip(axes.flat, panels):
        y = np.array([_f(r, key) for r in base], dtype=float)
        ax.plot(scale, y, '-', color='0.6', lw=1.5, zorder=1)
        ax.scatter(scale, y, c=colors, s=70, zorder=3, edgecolors='k', linewidths=0.5)
        if show_bl and bl is not None and _f(bl, key) is not None:
            ax.axhline(_f(bl, key), color=TYPE_COLOR['baseline'], ls='--', lw=1.6,
                       label='baseline (no filter)')
            ax.legend(fontsize=9)
        ax.axvline(1.0, color='k', ls=':', lw=1.2)          # accurate (assumed == actual)
        if key == 'min_h_r':
            ax.axhline(0.0, color='red', ls='--', lw=1.2)   # collision
        ax.set_title(title); ax.set_xlabel('assumed / actual responsiveness scale')
        ax.set_ylabel(ylab); ax.grid(alpha=0.3)

    # legend for the type colors (one shared, on the first axis)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', ls='', mec='k', mfc=c, ms=9, label=t)
               for t, c in TYPE_COLOR.items() if t != 'baseline']
    axes.flat[1].legend(handles=handles, fontsize=9, title='assumed model',
                        loc='best', ncol=2)

    fig.suptitle('Interaction-accuracy sweep: safety is robust, task performance degrades '
                 'with a conservative rear model', fontsize=15)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                      'accuracy_sweep.png')
    save_figure(fig, os.path.dirname(out) or '.', os.path.basename(out))


if __name__ == '__main__':
    main()
