"""
Comparison plotter for saved sandwiched three-car runs (run_sandwich.py).

The sandwiched scenario has a *forward* gap as well as a rear gap, so the rear-only layout of
plot_runs.py does not tell its story. This plotter overlays a set of sandwiched-HOCBF runs with
four stacked time series (left) plus a forward-vs-rear-gap phase portrait (right):

    panel 1 (top-left)    forward gap h_f vs time   (stays > 0 => forward-safe behind the lead)
    panel 2 (left)        rear gap    h_r vs time    (dips < 0 => rear-ended; dashed 0 and d_min)
    panel 3 (left)        ego velocity vs time       (per case; the shared lead profile is a
                                                      black solid line)
    panel 4 (left)        rear velocity vs time      (per case)
    panel 5 (left)        ego accel vs time          (solid = filtered, thin dashed = nominal)
    panel 6 (bot-left)    a_e_eff vs time            (the assumed rear's survivable brake fed to
                                                      the forward CBF; smaller => ego hangs back)
    phase A (top-right)   forward gap h_f vs ego speed v   (forward CBF margin vs speed)
    phase B (bot-right)   forward gap h_f vs rear gap h_r  (h_r<d_min and h_f<0 shaded unsafe;
                                                      only over-estimate crosses into h_r < 0)

I/O mirrors plot_runs.py exactly: a JSON spec with a ``runs`` map of ``{"label": "path"}`` plus
an optional ``output`` figure path (a bare ``{"label": "path"}`` map is also accepted). Relative
run paths resolve against the example dir, the spec's own dir, then cwd (first folder holding a
``series_*.npz`` wins). The ``--output`` flag overrides the spec's ``output``; otherwise the
figure is written next to the spec as ``compare_sandwich.png``.

    {
      "output": "output/compare_sandwich.png",
      "runs": {
        "passive (x0.0)":  "output/run_001",
        "under (x0.5)":    "output/run_002",
        "accurate (x1.0)": "output/run_003",
        "over (x2.5)":     "output/run_004"
      }
    }

Reference values (d_min, actual rear params, lead brake) are read from the first run's
config.json. Usage:

    uv run python examples/rear_aware/plot_sandwich_cases.py <spec.json> [--output out.png]
"""

import os
import json
import argparse

import numpy as np

from rear_aware_common import plt, LW, load_run_series, save_figure, _HERE   # noqa: E402
# Reuse plot_runs.py's spec/path/config-diff machinery so the I/O matches exactly.
from plot_runs import (PALETTE, resolve_run, find_series, print_config_diffs)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spec', help='path to the comparison spec JSON (see module docstring)')
    ap.add_argument('--output', default=None,
                    help="output figure path; overrides the spec's 'output' "
                         '(default: <spec_dir>/compare_sandwich.png)')
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict) or not spec:
        raise SystemExit('spec must be a non-empty JSON object')
    spec_dir = os.path.dirname(os.path.abspath(args.spec))
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
    rear = ref_cfg.get('rear_actual', {})
    lead_brake = (ref_cfg.get('lead') or {}).get('a_brake')

    # Report how the runs' configs differ (read straight from config.json).
    print_config_diffs(loaded)

    # ---- layout: 6 stacked time series (left) + 2 phase portraits (right) -----------
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(6, 2, width_ratios=[1.0, 1.2], hspace=0.4, wspace=0.2)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax_f)
    ax_v = fig.add_subplot(gs[2, 0], sharex=ax_f)
    ax_vr = fig.add_subplot(gs[3, 0], sharex=ax_f)
    ax_u = fig.add_subplot(gs[4, 0], sharex=ax_f)
    ax_a = fig.add_subplot(gs[5, 0], sharex=ax_f)
    ax_pv = fig.add_subplot(gs[0:3, 1])     # phase A: h_f vs ego speed
    ax_ph = fig.add_subplot(gs[3:6, 1])     # phase B: h_f vs rear gap

    # panel 1: forward gap h_f vs time
    for it in loaded:
        ax_f.plot(it['s']['t_ctrl'], it['s']['h_f'], color=it['color'], lw=LW, label=it['label'])
    ax_f.axhline(0.0, color='red', ls='--', lw=1.5, label='collision (h_f=0)')
    ax_f.set_ylabel('forward gap h_f [m]')
    ax_f.set_title('Forward gap to lead (stays > 0 => forward-safe)')
    ax_f.grid(alpha=0.3); ax_f.legend(fontsize=8, ncol=2)

    # panel 2: rear gap h_r vs time (label carries min h_r)
    for it in loaded:
        hr = it['s']['h_r']
        ax_r.plot(it['s']['t_ctrl'], hr, color=it['color'], lw=LW,
                  label=f"{it['label']}  (min={float(hr.min()):.2f})")
    ax_r.axhline(0.0, color='red', ls='--', lw=1.5, label='collision (h_r=0)')
    if d_min is not None:
        ax_r.axhline(d_min, color='k', ls='--', lw=1.5, label=f'd_min={d_min}')
    ax_r.set_ylabel('rear gap h_r [m]')
    ax_r.set_title('Rear gap (dips < 0 => rear-ended)')
    ax_r.grid(alpha=0.3); ax_r.legend(fontsize=8, ncol=2)

    # panel 3: ego velocity vs time (per case) + the shared lead profile (black solid)
    for it in loaded:
        ax_v.plot(it['s']['t_state'], it['s']['v_ego'], color=it['color'], lw=LW, label=it['label'])
    if loaded and 'v_lead' in loaded[0]['s']:
        ax_v.plot(loaded[0]['s']['t_state'], loaded[0]['s']['v_lead'], color='black', lw=LW,
                  label='lead')
    ax_v.set_ylabel('ego velocity [m/s]')
    ax_v.set_title('Ego velocity per case (lead = black solid)')
    ax_v.grid(alpha=0.3); ax_v.legend(fontsize=8, ncol=2)

    # panel 4: rear velocity vs time (per case)
    for it in loaded:
        ax_vr.plot(it['s']['t_state'], it['s']['v_rear'], color=it['color'], lw=LW,
                   label=it['label'])
    ax_vr.set_ylabel('rear velocity [m/s]')
    ax_vr.set_title('Rear follower velocity per case')
    ax_vr.grid(alpha=0.3); ax_vr.legend(fontsize=8, ncol=2)

    # panel 5: ego acceleration -- filtered (solid) and nominal (thin dashed, same colour)
    for it in loaded:
        ax_u.plot(it['s']['t_ctrl'], it['s']['u_ego'], color=it['color'], lw=LW, label=it['label'])
        ax_u.plot(it['s']['t_ctrl'], it['s']['u_nom'], color=it['color'], lw=LW * 0.6, ls='--')
    ax_u.set_ylabel('ego accel [m/s^2]')
    ax_u.set_title('Ego acceleration (solid = filtered, thin dashed = nominal)')
    ax_u.grid(alpha=0.3); ax_u.legend(fontsize=8, ncol=2)

    # panel 6: HOCBF -> rear-aware braking authority a_e_eff; backup CBF -> rollout min-h
    has_aee = any('a_e_eff' in it['s'] for it in loaded)
    if has_aee:
        for it in loaded:
            if 'a_e_eff' in it['s']:
                ax_a.plot(it['s']['t_ctrl'], it['s']['a_e_eff'], color=it['color'], lw=LW,
                          label=it['label'])
        ax_a.set_ylabel('a_e_eff [m/s^2]')
        ax_a.set_title('Rear-aware braking authority fed to the forward CBF '
                       '(smaller => ego hangs back more)')
    else:
        for it in loaded:
            if 'h_min' in it['s']:
                ax_a.plot(it['s']['t_ctrl'], it['s']['h_min'], color=it['color'], lw=LW,
                          label=it['label'])
        ax_a.axhline(0.0, color='red', ls='--', lw=1.2)
        ax_a.set_ylabel('backup min-h [m]')
        ax_a.set_title('Backup-rollout min barrier over the horizon (forward & rear)')
    ax_a.set_xlabel('time [s]')
    ax_a.grid(alpha=0.3); ax_a.legend(fontsize=8, ncol=2)

    # phase A: forward gap h_f (y) vs ego speed v (x) -------------------------------
    hf_all = np.concatenate([it['s']['h_f'] for it in loaded])
    y_lo, y_hi = min(hf_all.min(), 0.0) - 0.5, hf_all.max() * 1.1 + 0.5
    v_all = np.concatenate([it['s']['v_ego'] for it in loaded])
    vx_lo, vx_hi = min(v_all.min(), 0.0) - 0.3, v_all.max() * 1.05 + 0.5
    ax_pv.axhspan(y_lo, 0.0, color='orange', alpha=0.10, label='forward-unsafe (h_f < 0)')
    ax_pv.axhline(0.0, color='orange', ls='--', lw=1.5, label='lead collision (h_f=0)')
    for it in loaded:
        n = len(it['s']['h_f'])
        ax_pv.plot(it['s']['v_ego'][:n], it['s']['h_f'], color=it['color'], lw=LW, label=it['label'])
        ax_pv.plot(it['s']['v_ego'][0], it['s']['h_f'][0], 'o', color=it['color'], ms=6)
    ax_pv.set_xlim(vx_lo, vx_hi); ax_pv.set_ylim(y_lo, y_hi)
    ax_pv.set_xlabel('ego speed v [m/s]'); ax_pv.set_ylabel('forward gap h_f [m]')
    ax_pv.set_title('Phase portrait A: forward gap h_f vs ego speed v  (o = start)')
    ax_pv.grid(alpha=0.3); ax_pv.legend(fontsize=8, loc='best')

    # phase B: forward gap h_f (y) vs rear gap h_r (x) ------------------------------
    hr_all = np.concatenate([it['s']['h_r'] for it in loaded])
    x_lo, x_hi = min(hr_all.min(), 0.0) - 0.5, hr_all.max() * 1.1 + 0.5
    # unsafe regions: rear (h_r < d_min) and forward (h_f < 0)
    ax_ph.axvspan(x_lo, d_min if d_min is not None else 0.0, color='red', alpha=0.10,
                  label='rear-unsafe (h_r < d_min)')
    ax_ph.axhspan(y_lo, 0.0, color='orange', alpha=0.10, label='forward-unsafe (h_f < 0)')
    ax_ph.axvline(0.0, color='red', ls='--', lw=1.5, label='rear collision (h_r=0)')
    if d_min is not None:
        ax_ph.axvline(d_min, color='k', ls='--', lw=1.2, label=f'd_min={d_min}')
    ax_ph.axhline(0.0, color='orange', ls='--', lw=1.5, label='lead collision (h_f=0)')
    for it in loaded:
        ax_ph.plot(it['s']['h_r'], it['s']['h_f'], color=it['color'], lw=LW, label=it['label'])
        ax_ph.plot(it['s']['h_r'][0], it['s']['h_f'][0], 'o', color=it['color'], ms=6)
    ax_ph.set_xlim(x_lo, x_hi); ax_ph.set_ylim(y_lo, y_hi)
    ax_ph.set_xlabel('rear gap h_r [m]'); ax_ph.set_ylabel('forward gap h_f [m]')
    ax_ph.set_title('Phase portrait B: forward gap h_f vs rear gap h_r  (o = start)')
    ax_ph.grid(alpha=0.3); ax_ph.legend(fontsize=8, loc='best')

    bits = []
    if rear:
        bits.append(f"actual rear {ref_cfg.get('rear_model', 'ovm')} "
                    f"alpha/beta={rear.get('alpha')}/{rear.get('beta')}")
    if lead_brake is not None:
        bits.append(f'lead brake={lead_brake} m/s^2')
    sub = ('  (' + ', '.join(bits) + ')') if bits else ''
    fig.suptitle(f'Sandwiched three-car comparison ({len(loaded)} runs){sub}')

    if args.output:
        out = args.output if os.path.isabs(args.output) else os.path.join(os.getcwd(), args.output)
    elif spec_output:
        out = spec_output if os.path.isabs(spec_output) else os.path.join(_HERE, spec_output)
    else:
        out = os.path.join(spec_dir, 'compare_sandwich.png')
    out = os.path.normpath(out)
    save_figure(fig, os.path.dirname(out) or '.', os.path.basename(out))


if __name__ == '__main__':
    main()
