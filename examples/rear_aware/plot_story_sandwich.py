"""
Dedicated *story-line* plotter for the sandwiched three-car motivating example.

Where plot_sandwich_cases.py is the 8-panel debug view (every series + two phase portraits),
this plotter is the compact, paper-ready storyboard: a 2x3 grid that tells the "four ways to
handle the rear at one operating point" story at a glance.

    (0,0) ego speed  + lead profile (black)     (0,1) rear-follower speed       (0,2) case legend / text
    (1,0) forward gap h_f (to the lead)         (1,1) rear gap h_r (d_min line) (1,2) intervention effort bar

The effort bar is normalised to the largest case (tallest bar = 1.0) with the *actual*
∫|u-u_nom| dt printed on top. The top-right panel is a text block describing each case
(colour-coded), so the figure is self-contained for a paper/slide.

I/O mirrors plot_sandwich_cases.py: a JSON spec with a ``runs`` map of ``{"label": "path"}``
plus an optional ``output``. Run-path resolution and config-diff reporting are reused from
plot_runs.py. Per-case descriptions are derived from each run's config + realised metrics; a
spec may override them with a ``"notes": {"label": "text"}`` map.

    uv run python examples/rear_aware/plot_story_sandwich.py <spec.json> [--output out.png] [--no-latex]
"""

import os
import json
import argparse

import numpy as np

from rear_aware_common import plt, LW, load_run_series, _HERE   # noqa: E402
from plot_runs import (PALETTE, resolve_run, find_series, print_config_diffs)  # noqa: E402


def _set_paper_style():
    """Times + LaTeX rendering for paper-ready output (requires latex + newtx)."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',
        'axes.unicode_minus': False,
    })


def _set_plain_style():
    """Default matplotlib fonts (no LaTeX)."""
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'sans-serif',
        'axes.unicode_minus': True,
    })


def _verdict(hr_min, d_min):
    """One-word rear-safety verdict against the collision/keep-out thresholds."""
    if hr_min <= 0:
        return 'rear-end'
    if d_min is not None and hr_min < d_min:
        return 'below d_min'
    return 'safe'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spec', help='path to the comparison spec JSON (see module docstring)')
    ap.add_argument('--output', default=None,
                    help="output figure path; overrides the spec's 'output' "
                         '(default: <spec_dir>/story_sandwich.png)')
    ap.add_argument('--no-latex', action='store_true',
                    help='disable LaTeX/Times rendering (use default matplotlib fonts)')
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict) or not spec:
        raise SystemExit('spec must be a non-empty JSON object')
    spec_dir = os.path.dirname(os.path.abspath(args.spec))
    bases = [_HERE, spec_dir, os.getcwd()]

    runs = spec['runs'] if 'runs' in spec else spec
    if not isinstance(runs, dict) or not runs:
        raise SystemExit('spec must contain a non-empty "runs" map of {"run-name": "path"}')
    spec_output = spec.get('output') if 'runs' in spec else None
    notes = spec.get('notes', {}) if 'runs' in spec else {}

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

    print_config_diffs(loaded)

    tex = not args.no_latex
    _set_paper_style() if tex else _set_plain_style()
    S = 1.2 * 1.2 * 1.1    # 1.2x base, then a further 1.1x for the compact proposal figure
    LBL_FS, TTL_FS, LEG_FS, NUM_FS = 14 * S, 14 * S, 12 * S, 12 * S
    TXT_FS = 8.0 * 1.1    # small on purpose (room for manual overlay), matched 1.1x bump

    # Mode-aware math/label fragments.
    HF = r'$h_{\rm f}$' if tex else 'h_f'
    HR = r'$h_{\rm r}$' if tex else 'h_r'
    INT = r'$\int\!\left|u-u_{\mathrm{nom}}\right|\,\mathrm{d}t$' if tex else '∫|u-u_nom| dt'
    MS = r'\,' if tex else ' '  # spacing inside math is cosmetic only when tex

    # ---- per-case realised metrics ------------------------------------------------
    for it in loaded:
        s = it['s']
        u, un, t = s['u_ego'], s['u_nom'], s['t_ctrl']
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        it['effort'] = float(np.sum(np.abs(u - un)) * dt)
        it['hf_min'] = float(s['h_f'].min())
        it['hr_min'] = float(s['h_r'].min())
        it['verdict'] = _verdict(it['hr_min'], d_min)

    # ---- layout: 3 rows x 2 cols --------------------------------------------------
    #   row 1: case text | effort bar    row 2: ego speed | rear speed    row 3: h_f | h_r
    # Compact proposal layout: rows 2 & 3 shrunk to 0.8 of row 1 -> figure height
    # scaled to (1+0.8+0.8)/3 of the original 14 in.
    fig = plt.figure(figsize=(13, 14 * (2.6 / 3.0)))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 0.8], hspace=0.30, wspace=0.20)
    ax_txt = fig.add_subplot(gs[0, 0])    # case legend / text
    ax_bar = fig.add_subplot(gs[0, 1])    # effort bar
    ax_ve = fig.add_subplot(gs[1, 0])     # ego speed + lead
    ax_vr = fig.add_subplot(gs[1, 1], sharex=ax_ve)   # rear speed
    ax_hf = fig.add_subplot(gs[2, 0], sharex=ax_ve)   # forward gap
    ax_hr = fig.add_subplot(gs[2, 1], sharex=ax_ve)   # rear gap

    # (0,0) ego speed per case + shared lead profile (black)
    for it in loaded:
        ax_ve.plot(it['s']['t_state'], it['s']['v_ego'], color=it['color'], lw=LW, label=it['label'])
    if loaded and 'v_lead' in loaded[0]['s']:
        ax_ve.plot(loaded[0]['s']['t_state'], loaded[0]['s']['v_lead'], color='black',
                   lw=LW * 0.9, ls='-', label='lead')
    ax_ve.set_title('Ego speed (lead = black)', fontsize=TTL_FS)
    ax_ve.grid(alpha=0.3); ax_ve.tick_params(labelsize=LEG_FS)

    # (0,1) rear-follower speed per case
    for it in loaded:
        ax_vr.plot(it['s']['t_state'], it['s']['v_rear'], color=it['color'], lw=LW)
    ax_vr.set_title('Rear speed', fontsize=TTL_FS)
    ax_vr.grid(alpha=0.3); ax_vr.tick_params(labelsize=LEG_FS)

    # (1,0) forward gap h_f vs time
    for it in loaded:
        ax_hf.plot(it['s']['t_ctrl'], it['s']['h_f'], color=it['color'], lw=LW)
    ax_hf.axhline(0.0, color='red', ls='--', lw=1.5)
    ax_hf.set_xlabel('time [s]', fontsize=LBL_FS)
    gt0 = '($>0$ = no hit)' if tex else '(>0 = no hit)'
    ax_hf.set_title(f'Ego headway {HF} {gt0}', fontsize=TTL_FS)
    ax_hf.grid(alpha=0.3); ax_hf.tick_params(labelsize=LEG_FS)

    # (1,1) rear gap h_r vs time with collision + d_min lines (labelled inline)
    for it in loaded:
        ax_hr.plot(it['s']['t_ctrl'], it['s']['h_r'], color=it['color'], lw=LW)
    ax_hr.axhline(0.0, color='red', ls='--', lw=1.5)
    if d_min is not None:
        ax_hr.axhline(d_min, color='k', ls='--', lw=1.5)
    ax_hr.set_xlabel('time [s]', fontsize=LBL_FS)
    ax_hr.set_title(f'Rear headway {HR} (keep above $d_{{\\min}}$)' if tex
                    else 'Rear headway h_r (keep above d_min)', fontsize=TTL_FS)
    ax_hr.grid(alpha=0.3); ax_hr.tick_params(labelsize=LEG_FS)

    # Shared y-limits: ego/rear speed on one scale, h_f/h_r distance on another.
    def _pad(lo, hi, frac=0.06):
        m = (hi - lo) * frac or 0.5
        return lo - m, hi + m

    v_arrs = [it['s']['v_ego'] for it in loaded] + [it['s']['v_rear'] for it in loaded]
    if loaded and 'v_lead' in loaded[0]['s']:
        v_arrs.append(loaded[0]['s']['v_lead'])
    v_lo, v_hi = _pad(min(a.min() for a in v_arrs), max(a.max() for a in v_arrs))
    ax_ve.set_ylim(v_lo, v_hi); ax_vr.set_ylim(v_lo, v_hi)

    d_arrs = ([it['s']['h_f'] for it in loaded] + [it['s']['h_r'] for it in loaded])
    d_lo = min(0.0, min(a.min() for a in d_arrs))
    d_hi = max(a.max() for a in d_arrs)
    d_lo, d_hi = _pad(d_lo, d_hi)
    ax_hf.set_ylim(d_lo, d_hi); ax_hr.set_ylim(d_lo, d_hi)

    # Compact unit label: between the top two y-ticks, snug to the axis (saves width).
    def _unit_at_top(ax, unit):
        lo, hi = ax.get_ylim()
        ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
        yv = 0.5 * (ticks[-1] + ticks[-2]) if len(ticks) >= 2 else lo + 0.85 * (hi - lo)
        ax.text(-0.015, yv, unit, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=LBL_FS)

    for ax in (ax_ve, ax_vr):
        _unit_at_top(ax, '[m/s]')
    for ax in (ax_hf, ax_hr):
        _unit_at_top(ax, '[m]')

    # Shade the collision zone (gap < 0) and label the reference lines inline.
    drange = d_hi - d_lo
    ax_hf.axhspan(d_lo, 0.0, color='red', alpha=0.08, zorder=0)
    ax_hr.axhspan(d_lo, 0.0, color='red', alpha=0.08, zorder=0)
    ax_hf.text(1.0, -0.03 * drange,
               ('collision ($h_{\\rm f}=0$)' if tex else 'collision (h_f=0)'),
               color='red', fontsize=LEG_FS, va='top', ha='left')
    if d_min is not None:
        ax_hr.text(1.0, d_min + 0.03 * drange,
                   (rf'$d_{{\min}}={d_min:g}$' if tex else f'd_min={d_min:g}'),
                   color='k', fontsize=LEG_FS, va='bottom', ha='left')
    ax_hr.text(1.0, -0.03 * drange,
               ('collision ($h_{\\rm r}=0$)' if tex else 'collision (h_r=0)'),
               color='red', fontsize=LEG_FS, va='top', ha='left')

    # Initial rear gap (all cases share the initial condition) -- echoed in the footer.
    hr0 = float(loaded[0]['s']['h_r'][0]) if loaded else None

    # (1,2) intervention effort bar -- normalised to the largest, actual value on top
    labels = [chr(ord('1') + i) for i in range(len(loaded))]  # short tick labels 1..n
    efforts = np.array([it['effort'] for it in loaded])
    colors = [it['color'] for it in loaded]
    e_max = float(efforts.max()) if efforts.size and efforts.max() > 0 else 1.0
    norm = efforts / e_max
    xs = np.arange(len(loaded))
    ax_bar.bar(xs, norm, color=colors, alpha=0.85)
    for x, n, e in zip(xs, norm, efforts):
        ax_bar.text(x, n + 0.02, f'{e:.1f}', ha='center', va='bottom',
                    fontsize=NUM_FS, fontweight='bold')
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(labels, fontsize=LEG_FS)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel('normalised to max', fontsize=LBL_FS)
    ax_bar.set_title(f'Intervention effort {INT}', fontsize=TTL_FS)
    ax_bar.tick_params(labelsize=LEG_FS)
    ax_bar.grid(alpha=0.3, axis='y')

    # (0,0) text panel: colour-coded case descriptions (small; room for manual overlay)
    ax_txt.axis('off')
    ax_txt.set_title('Cases', fontsize=TXT_FS * 1.25, loc='left')
    y = 0.96
    for i, it in enumerate(loaded):
        tag = labels[i]
        head = f'{tag}.  {it["label"]}'
        # derived one-liner: min h_r / verdict, min h_f, effort
        sub = (rf'min {HR}={it["hr_min"]:.2f} ({it["verdict"]}),  '
               rf'min {HF}={it["hf_min"]:.2f},  effort={it["effort"]:.1f}')
        override = notes.get(it['label'])
        ax_txt.text(0.0, y, head, transform=ax_txt.transAxes, fontsize=TXT_FS,
                    color=it['color'], fontweight='bold', va='top')
        y -= 0.055
        ax_txt.text(0.04, y, override if override else sub, transform=ax_txt.transAxes,
                    fontsize=TXT_FS * 0.85, color='0.25', va='top')
        y -= 0.085
    # scenario footer inside the text panel
    foot = []
    if hr0 is not None:
        foot.append((rf'initial rear gap $h_{{\rm r}}(0)={hr0:.1f}$ m ($d_{{\min}}={d_min:g}$)'
                     if tex else f'initial rear gap h_r(0)={hr0:.1f} m (d_min={d_min:g})'))
    if rear:
        foot.append((rf'actual rear $\alpha/\beta={rear.get("alpha")}/{rear.get("beta")}$'
                     if tex else f'actual rear a/b={rear.get("alpha")}/{rear.get("beta")}'))
    if lead_brake is not None:
        foot.append((rf'lead brake ${lead_brake}\,$m/s$^2$' if tex
                     else f'lead brake {lead_brake} m/s^2'))
    if foot:
        ax_txt.text(0.0, max(y, 0.06), '\n'.join(foot), transform=ax_txt.transAxes,
                    fontsize=TXT_FS * 0.78, color='0.4', va='top', style='italic')

    if args.output:
        out = args.output if os.path.isabs(args.output) else os.path.join(os.getcwd(), args.output)
    elif spec_output:
        out = spec_output if os.path.isabs(spec_output) else os.path.join(_HERE, spec_output)
    else:
        out = os.path.join(spec_dir, 'story_sandwich.png')
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved figure to {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
