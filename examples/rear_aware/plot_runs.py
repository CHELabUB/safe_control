"""
Comparison plotter for saved rear-aware runs.

Each run folder (one ``--method`` per run) holds a single ``series_<method>.npz`` plus a
``config.json`` (see rear_aware_common.save_run_series). This script overlays a set of
those runs in a fixed layout, selected through a minimal JSON spec.

Spec JSON: a nested object with a ``runs`` map of ``{"run-name": "path"}`` (each key is the
label, each value its run folder) and an optional ``output`` figure path. A bare
``{"run-name": "path"}`` object (no ``runs`` key) is still accepted as the runs map.

Relative run paths are resolved by trying, in order, the example directory
(``examples/rear_aware``), the spec file's own directory, then the current directory — the
first that actually contains a ``series_*.npz`` wins. So paths written relative to the
example dir (e.g. ``saved_results/ego_rear/run_001``) or to the spec file (e.g. ``run_001``
sitting beside it) both work.

    {
      "output": "output/compare.png",
      "runs": {
        "baseline": "output/run_001",
        "backup CBF": "output/run_002",
        "HOCBF": "output/run_003"
      }
    }

The output path is the spec's ``output`` if present, overridden by ``--output`` when
supplied, else ``<spec_dir>/compare_runs.png``.

Layout (left column = time series, right = phase portrait):
    panel 1 (top-left)    rear gap h_r vs time      + dashed h_r=0 and h_r=d_min
    panel 2 (mid-left)    ego velocity vs time      + dashed target speed
    panel 3 (bot-left)    ego acceleration vs time  (solid=actual, thin dashed=nominal)
    phase  (right)        ego speed v vs rear gap h_r  + desired-speed line, h=0, h=d_min
                          (h<d_min shaded red) and the OVM & IDM range-policy curves

Reference values (d_min, target speed, rear params for the policy curves) are read from
the first run's config.json. Usage:

    uv run python examples/rear_aware/plot_runs.py <spec.json> [--output out.png]
"""

import os
import json
import glob
import argparse

import numpy as np

from rear_aware_common import plt, LW, load_run_series, save_figure, _HERE   # noqa: E402

PALETTE = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange',
           'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray']


def resolve_run(rel, bases):
    """Resolve a (possibly relative) run path to a folder containing series_*.npz.

    A relative path is tried against each candidate base (example dir, spec dir, cwd) so
    specs written relative to the example directory *or* to the spec file both work. The
    first candidate that actually holds a series file wins.
    """
    cands = [rel] if os.path.isabs(rel) else [os.path.normpath(os.path.join(b, rel))
                                              for b in bases]
    for c in cands:
        if glob.glob(os.path.join(c, 'series_*.npz')):
            return c
    raise SystemExit("no run with series_*.npz found for '%s'; tried:\n  %s"
                     % (rel, "\n  ".join(cands)))


def find_series(run_path):
    """Locate the single series_*.npz in a run folder (one method per run)."""
    files = sorted(glob.glob(os.path.join(run_path, 'series_*.npz')))
    if not files:
        raise FileNotFoundError(f"no series_*.npz in {run_path}")
    return files[0]


def ovm_policy_v(rear, h):
    """OVM optimal-velocity V(h) = clip(kappa*(h-h_st), 0, v_max)."""
    return np.clip(rear['kappa'] * (np.asarray(h) - rear['h_st']), 0.0, rear['v_max'])


def idm_policy_h(rear, v):
    """IDM equilibrium gap for speed v: s = (s0 + v*T)/sqrt(1-(v/v0)^4)."""
    v0, T, s0 = rear['v0'], rear['T'], rear['s0']
    vv = np.clip(np.asarray(v), 0.0, v0 * 0.999)
    return (s0 + vv * T) / np.sqrt(1.0 - (vv / v0) ** 4)


_MISSING = object()


def _flatten_cfg(cfg, prefix=''):
    """Flatten a (nested) config dict to dotted keys, e.g. rear_actual.kappa."""
    flat = {}
    for k, v in cfg.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            flat.update(_flatten_cfg(v, key + '.'))
        else:
            flat[key] = v
    return flat


def print_config_diffs(loaded):
    """Print the config keys that differ across the compared runs (from config.json).

    A key absent in a run's (pruned) config is shown as '-'; keys identical across all
    runs are omitted. This surfaces every difference, not just the register columns.
    """
    flats = [_flatten_cfg(it.get('cfg', {})) for it in loaded]
    labels = [it['label'] for it in loaded]
    keys = sorted(set().union(*[set(f) for f in flats]) if flats else set())
    diff_keys = [k for k in keys if any(f.get(k, _MISSING) != flats[0].get(k, _MISSING)
                                        for f in flats)]
    if not diff_keys:
        print(f'\nConfigs identical across {len(labels)} runs.')
        return

    def fmt(v):
        if v is _MISSING:
            return '-'
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f'{v:g}'
        return str(v)

    cells = {k: [fmt(f.get(k, _MISSING)) for f in flats] for k in diff_keys}
    kw = max([len('config key')] + [len(k) for k in diff_keys])
    cw = [max(len(labels[j]), *(len(cells[k][j]) for k in diff_keys)) for j in range(len(labels))]
    print(f'\nConfig differences across {len(labels)} runs '
          f'(identical keys hidden; "-" = not in that run):')
    print('  ' + 'config key'.ljust(kw) + '   '
          + '   '.join(labels[j].ljust(cw[j]) for j in range(len(labels))))
    for k in diff_keys:
        print('  ' + k.ljust(kw) + '   '
              + '   '.join(cells[k][j].ljust(cw[j]) for j in range(len(labels))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spec', help='path to the comparison spec JSON (see module docstring)')
    ap.add_argument('--output', default=None,
                    help="output figure path; overrides the spec's 'output' "
                         '(default: <spec_dir>/compare_runs.png)')
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict) or not spec:
        raise SystemExit('spec must be a non-empty JSON object')
    spec_dir = os.path.dirname(os.path.abspath(args.spec))
    # Bases tried for relative paths: example dir, the spec's own dir, then cwd.
    bases = [_HERE, spec_dir, os.getcwd()]

    # Nested form ({"runs": {...}, "output": ...}) or the bare {"label": "path"} map.
    runs = spec['runs'] if 'runs' in spec else spec
    if not isinstance(runs, dict) or not runs:
        raise SystemExit('spec must contain a non-empty "runs" map of {"run-name": "path"}')
    spec_output = spec.get('output') if 'runs' in spec else None

    loaded, ref_cfg = [], None
    for i, (label, rel) in enumerate(runs.items()):
        path = resolve_run(rel, bases)
        series = load_run_series(find_series(path))
        cfg_path = os.path.join(path, 'config.json')
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        loaded.append({'label': label, 's': series, 'cfg': cfg,
                       'color': PALETTE[i % len(PALETTE)]})
        if ref_cfg is None and cfg:
            ref_cfg = cfg
    ref_cfg = ref_cfg or {}
    d_min = ref_cfg.get('d_min')
    v_des = ref_cfg.get('v_desired')                # None in stop mode
    rear = ref_cfg.get('rear_actual', {})

    # Report how the runs' configs differ (read straight from config.json, which records
    # the full pruned config -- register.csv only keeps a few key columns).
    print_config_diffs(loaded)

    # ---- layout: 3 stacked time-series (left) + phase portrait (right) -------------
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.2], hspace=0.32, wspace=0.2)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_h)
    ax_a = fig.add_subplot(gs[2, 0], sharex=ax_h)
    ax_ph = fig.add_subplot(gs[:, 1])

    # panel 1: rear gap h_r vs time
    for it in loaded:
        ax_h.plot(it['s']['t_ctrl'], it['s']['h_r'], color=it['color'], lw=LW, label=it['label'])
    ax_h.axhline(0.0, color='red', ls='--', lw=1.5, label='collision (h_r=0)')
    if d_min is not None:
        ax_h.axhline(d_min, color='k', ls='--', lw=1.5, label=f'd_min={d_min}')
    ax_h.set_ylabel('rear gap h_r [m]'); ax_h.set_title('Rear gap')
    ax_h.grid(alpha=0.3); ax_h.legend(fontsize=8, ncol=2)

    # panel 2: ego velocity vs time
    for it in loaded:
        ax_v.plot(it['s']['t_state'], it['s']['v_ego'], color=it['color'], lw=LW, label=it['label'])
    if v_des is not None:
        ax_v.axhline(v_des, color='k', ls='--', lw=1.5, label=f'target speed={v_des}')
    ax_v.set_ylabel('ego velocity [m/s]'); ax_v.set_title('Velocity')
    ax_v.grid(alpha=0.3); ax_v.legend(fontsize=8, ncol=2)

    # panel 3: ego acceleration -- actual (solid) and nominal (thin dashed, same colour)
    for it in loaded:
        ax_a.plot(it['s']['t_ctrl'], it['s']['u_ego'], color=it['color'], lw=LW, label=it['label'])
        ax_a.plot(it['s']['t_ctrl'], it['s']['u_nom'], color=it['color'], lw=LW * 0.6, ls='--')
    ax_a.set_ylabel('ego accel [m/s^2]'); ax_a.set_xlabel('time [s]')
    ax_a.set_title('Ego acceleration (solid = actual, thin dashed = nominal)')
    ax_a.grid(alpha=0.3); ax_a.legend(fontsize=8, ncol=2)

    # phase portrait: v vs h_r (x = rear gap h_r, y = ego speed v) -------------------
    h_all = np.concatenate([it['s']['h_r'] for it in loaded])
    v_all = np.concatenate([it['s']['v_ego'] for it in loaded])
    v_hi = max(v_all.max(), v_des or 0.0, rear.get('v0', 0.0)) * 1.05 + 1e-6
    h_hi = max(h_all.max(), (d_min or 0.0)) * 1.15 + 0.5
    h_lo = min(h_all.min(), 0.0) - 0.5

    # safety-violation region (h < d_min) shaded red
    if d_min is not None:
        ax_ph.axvspan(h_lo, d_min, color='red', alpha=0.12, label='unsafe (h < d_min)')
    # range-policy curves: OVM and IDM (rear params from the reference config)
    if rear:
        h_grid = np.linspace(max(h_lo, 0.0), h_hi, 240)
        ax_ph.plot(h_grid, ovm_policy_v(rear, h_grid), color='black', ls='--', lw=1.4,
                   label='OVM range policy')
        v_grid = np.linspace(0.0, rear.get('v0', v_hi) * 0.999, 240)
        h_idm = idm_policy_h(rear, v_grid)
        m = h_idm <= h_hi
        ax_ph.plot(h_idm[m], v_grid[m], color='teal', ls='-.', lw=1.4, label='IDM range policy')
    # collision / safety / desired-speed reference lines
    ax_ph.axvline(0.0, color='red', ls='--', lw=1.5, label='collision (h=0)')
    if d_min is not None:
        ax_ph.axvline(d_min, color='k', ls='--', lw=1.5, label=f'd_min={d_min}')
    if v_des is not None:
        ax_ph.axhline(v_des, color='gray', ls='--', lw=1.5, label=f'desired speed={v_des}')
    # trajectories (h_r, v_ego) -- align lengths (h_r is one shorter than v_ego)
    for it in loaded:
        n = len(it['s']['h_r'])
        ax_ph.plot(it['s']['h_r'], it['s']['v_ego'][:n], color=it['color'], lw=LW, label=it['label'])
    ax_ph.set_xlim(h_lo, h_hi); ax_ph.set_ylim(0.0, v_hi)
    ax_ph.set_xlabel('rear gap h_r [m]'); ax_ph.set_ylabel('ego speed v [m/s]')
    ax_ph.set_title('Phase portrait: ego speed v vs rear gap h_r')
    ax_ph.grid(alpha=0.3); ax_ph.legend(fontsize=8, loc='best')

    fig.suptitle(f'Rear-aware comparison ({len(loaded)} runs)')
    # --output (relative to cwd) overrides the spec's "output" (relative to the example
    # dir, matching the run paths), which overrides the default next to the spec.
    if args.output:
        out = args.output if os.path.isabs(args.output) else os.path.join(os.getcwd(), args.output)
    elif spec_output:
        out = spec_output if os.path.isabs(spec_output) else os.path.join(_HERE, spec_output)
    else:
        out = os.path.join(spec_dir, 'compare_runs.png')
    out = os.path.normpath(out)
    save_figure(fig, os.path.dirname(out) or '.', os.path.basename(out))


if __name__ == '__main__':
    main()
