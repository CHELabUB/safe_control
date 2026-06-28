"""
Monte-Carlo stress test for interaction handling in the sandwiched backup CBF.

Randomizes the REAL rear OVM gains (alpha, beta) around a nominal and runs four ways of
handling the rear interaction, to show they cluster in different regions of the
(intervention-effort, min h_f, min h_r) space:

  A  worst-case baseline   assumed rear passive (0,0), plain controller        -> safe, conservative
  B  nominal, no buffer    assumed rear nominal (0.32,0.224), plain            -> cheap, unsafe on
                                                                                  sluggish draws
  C  nominal + ISSf buffer  assumed nominal + buffer L_inter*|nom-actual|+res  -> safe, moderate
  D  oracle + buffer        assumed = ACTUAL (per draw) + buffer (->residue)   -> safe, least
                                                                                  conservative (gold)

All four see the SAME paired random draws (fixed seed). Metrics are the REALIZED gap minima
(out['h_f'].min(), out['h_r'].min()) -- NOT the backup-rollout preview -- and the control
intervention integral. Reuses run_sandwich.build_parser/build_cfg/simulate_sandwich (no edits to
the controller or driver). No RunRegistry: 80 sims would otherwise litter run folders.

Three artifacts, raw and aggregated saved separately; the figure is built from the aggregated json
ALONE (re-plot without recomputing):
  output/sandwich_bcbf_stress/raw/stress_raw.csv
  output/sandwich_bcbf_stress/aggregated/stress_aggregated.json
  output/sandwich_bcbf_stress/stress_clusters.png

Usage:
  uv run python examples/rear_aware/stress_test_sandwich.py --pilot      # deterministic separation check
  uv run python examples/rear_aware/stress_test_sandwich.py              # 20-draw MC (default)
  uv run python examples/rear_aware/stress_test_sandwich.py --from-raw <csv>          # re-aggregate + plot
  uv run python examples/rear_aware/stress_test_sandwich.py --from-aggregated <json>  # re-plot only
"""

import os
import sys
import csv
import json
import argparse
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, '..', 'car_following'),
           os.path.join(_HERE, '..', 'double_integrator'), os.path.join(_HERE, '..')):
    sys.path.insert(0, os.path.abspath(_p))

import run_sandwich as RS                                    # noqa: E402
from rear_aware_common import plt, save_figure               # noqa: E402

NOM = (0.32, 0.224)                                          # nominal rear (alpha, beta)

# group -> (assumed-rear spec, uses-ISSf).  assumed: 'passive' | 'nominal' | 'oracle'
GROUPS = {
    'A': ('passive', False),
    'B': ('nominal', False),
    'C': ('nominal', True),
    'D': ('oracle',  True),
}
GROUP_LABELS = {
    'A': 'A: worst-case (passive)',
    'B': 'B: nominal, no buffer',
    'C': 'C: nominal + ISSf buffer',
    'D': 'D: oracle + buffer',
}
GROUP_COLORS = {'A': 'tab:gray', 'B': 'tab:red', 'C': 'tab:blue', 'D': 'tab:green'}
ORDER = ['A', 'B', 'C', 'D']


def build_argv(group, a, b, scn, issf):
    """Construct the run_sandwich CLI argv for one (group, actual rear) combination."""
    assumed_kind, use_issf = GROUPS[group]
    argv = ['--method', 'sandwiched_bcbf',
            '--rear-kappa', str(scn['rear_kappa']), '--gap-r0', str(scn['gap_r0']),
            '--backup-beta-f', str(scn['backup_beta_f']),
            '--lead-a-brake', str(scn['lead_a_brake']),
            '--a-lead-decel', str(scn['lead_a_brake']),              # forward CBF a_l matches lead
            '--rear-alpha', str(a), '--rear-beta', str(b)]           # the ACTUAL rear
    if assumed_kind == 'passive':
        argv += ['--assumed-rear-alpha', '0', '--assumed-rear-beta', '0']
    elif assumed_kind == 'nominal':
        argv += ['--assumed-rear-alpha', str(NOM[0]), '--assumed-rear-beta', str(NOM[1])]
    else:                                                            # oracle: belief = actual
        argv += ['--assumed-rear-alpha', str(a), '--assumed-rear-beta', str(b)]
    if use_issf:
        argv += ['--l-inter-ratio', str(issf['l_inter_ratio']),
                 '--l-inter-residue', str(issf['l_inter_residue']),
                 '--rho-rear', str(issf['rho_rear']), '--gamma', str(issf['gamma'])]
    return argv


def run_one(group, a, b, scn, issf):
    """Run one sim; return realized-metric dict (no disk writes)."""
    args = RS.build_parser().parse_args(build_argv(group, a, b, scn, issf))
    cfg = RS.build_cfg(args, 'sandwiched_bcbf')
    dt = args.dt
    n_sim = int(round(args.tf / dt))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        out = RS.simulate_sandwich(cfg, dt, n_sim, 'sandwiched_bcbf')
    min_hf = float(out['h_f'].min())
    min_hr = float(out['h_r'].min())
    effort = float(np.sum(np.abs(out['u_ego'] - out['u_nom'])) * dt)
    return {'group': group, 'alpha': float(a), 'beta': float(b),
            'min_h_f': min_hf, 'min_h_r': min_hr, 'effort': effort,
            'rear_buffer': float(cfg.get('rear_buffer', 0.0)),
            'rear_end': int(min_hr <= 0.0), 'front_collision': int(min_hf <= 0.0),
            'clears_dmin': int(min_hr > cfg['d_min']), 'd_min': float(cfg['d_min'])}


def draw_actuals(n, seed, frac):
    rng = np.random.default_rng(seed)
    lo = (NOM[0] * (1 - frac), NOM[1] * (1 - frac))
    hi = (NOM[0] * (1 + frac), NOM[1] * (1 + frac))
    return [(float(rng.uniform(lo[0], hi[0])), float(rng.uniform(lo[1], hi[1]))) for _ in range(n)]


RAW_FIELDS = ['group', 'draw', 'alpha', 'beta', 'min_h_f', 'min_h_r', 'effort',
              'rear_buffer', 'rear_end', 'front_collision', 'clears_dmin']


def _mc_task(task, scn, issf):
    """Worker: run one (draw-index, group, alpha, beta) combo (picklable for Pool)."""
    di, group, a, b = task
    r = run_one(group, a, b, scn, issf)
    r['draw'] = di
    return r


def run_mc(scn, issf, n, seed, frac, jobs=1):
    """Run the paired MC over all groups; return list of raw rows. jobs>1 -> multiprocessing."""
    draws = draw_actuals(n, seed, frac)
    tasks = [(di, g, a, b) for di, (a, b) in enumerate(draws) for g in ORDER]
    total = len(tasks)
    rows = []
    if jobs and jobs > 1:
        from multiprocessing import Pool
        from functools import partial
        worker = partial(_mc_task, scn=scn, issf=issf)
        with Pool(processes=jobs) as pool:
            for k, r in enumerate(pool.imap_unordered(worker, tasks), 1):
                rows.append(r)
                if k % 10 == 0 or k == total:
                    print(f"  [{k:3d}/{total}] done", flush=True)
        rows.sort(key=lambda r: (int(r['draw']), ORDER.index(r['group'])))
    else:
        for k, task in enumerate(tasks, 1):
            r = _mc_task(task, scn, issf)
            rows.append(r)
            di, _, a, b = task
            print(f"  [{k:3d}/{total}] draw {di:2d} (a={a:.3f},b={b:.3f}) {r['group']}: "
                  f"min_h_f={r['min_h_f']:6.2f} min_h_r={r['min_h_r']:6.2f} "
                  f"effort={r['effort']:6.2f} buf={r['rear_buffer']:.2f}", flush=True)
    return rows


def run_pilot(scn, issf, scales):
    """Deterministic actual = scale*nominal sweep across groups; print + return rows."""
    rows = []
    print(f"{'scale':>6} {'group':>6} {'min_h_f':>8} {'min_h_r':>8} {'effort':>8} {'buf':>6}")
    for s in scales:
        a, b = NOM[0] * s, NOM[1] * s
        for group in ORDER:
            r = run_one(group, a, b, scn, issf)
            r['draw'] = s
            rows.append(r)
            print(f"{s:6.2f} {group:>6} {r['min_h_f']:8.2f} {r['min_h_r']:8.2f} "
                  f"{r['effort']:8.2f} {r['rear_buffer']:6.2f}", flush=True)
    return rows


def write_raw(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=RAW_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote raw -> {path}")


def read_raw(path):
    with open(path) as fh:
        return [{k: (v if k == 'group' else float(v)) for k, v in row.items()}
                for row in csv.DictReader(fh)]


def aggregate(rows, meta):
    """Per-group summary (means/stds + covariances for ellipses + safety rates)."""
    d_min = float(rows[0]['d_min']) if 'd_min' in rows[0] else meta.get('d_min', 1.0)
    groups = {}
    for group in ORDER:
        g = [r for r in rows if r['group'] == group]
        if not g:
            continue
        eff = np.array([r['effort'] for r in g])
        hf = np.array([r['min_h_f'] for r in g])
        hr = np.array([r['min_h_r'] for r in g])

        def mstd(x):
            return {'mean': float(x.mean()), 'std': float(x.std(ddof=1) if x.size > 1 else 0.0),
                    'min': float(x.min()), 'max': float(x.max())}

        cov_ehr = np.cov(eff, hr) if eff.size > 1 else np.zeros((2, 2))
        cov_ehf = np.cov(eff, hf) if eff.size > 1 else np.zeros((2, 2))
        groups[group] = {
            'label': GROUP_LABELS[group], 'n': len(g),
            'effort': mstd(eff), 'min_h_f': mstd(hf), 'min_h_r': mstd(hr),
            'cov_effort_hr': cov_ehr.tolist(), 'cov_effort_hf': cov_ehf.tolist(),
            'pct_collision_free': float(100.0 * np.mean(hr > 0)),
            'pct_clears_dmin': float(100.0 * np.mean(hr > d_min)),
            'pct_front_safe': float(100.0 * np.mean(hf > 0)),
        }
    return {'meta': meta, 'd_min': d_min, 'groups': groups}


def _ellipse(ax, mean, cov, color, nsig=1.0):
    from matplotlib.patches import Ellipse
    cov = np.array(cov)
    if not np.all(np.isfinite(cov)) or np.allclose(cov, 0):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nsig * np.sqrt(vals)
    e = Ellipse(xy=mean, width=w, height=h, angle=angle, facecolor=color, alpha=0.15,
                edgecolor=color, lw=1.5)
    ax.add_patch(e)


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
    """Default matplotlib fonts (no LaTeX) -- the pre-paper styling."""
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'sans-serif',
        'axes.unicode_minus': True,
    })


def plot_aggregated(agg, out_png, use_tex=True):
    from matplotlib.lines import Line2D
    tex = bool(use_tex)
    _set_paper_style() if tex else _set_plain_style()
    groups = agg['groups']
    d_min = agg['d_min']
    S = 1.2                                              # font scale for paper use
    LEG_FS, NUM_FS, LBL_FS, TTL_FS = 15 * S, 15 * S, 13 * S, 15 * S
    INL_FS, HDR_FS, SUP_FS = 11 * S, 16 * S, 17 * S      # inline / header / suptitle

    # Mode-aware label fragments: LaTeX/Times (tex) vs plain unicode (else).
    INT = r'$\int\!\left|u-u_{\mathrm{nom}}\right|\,\mathrm{d}t$' if tex else '∫|u-u_nom| dt'
    HF = r'$h_f$' if tex else 'h_f'
    HR = r'$h_r$' if tex else 'h_r'
    SIGMA = r'$1\sigma$' if tex else '1σ'
    PM = r'$\pm$' if tex else '±'
    PCT = r'\%' if tex else '%'
    DASH = '--' if tex else '—'
    DMIN_LBL = (r'$d_{\min}=%.1f$' % d_min) if tex else f'd_min={d_min}'

    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.27, wspace=0.17,
                          top=0.80, bottom=0.06, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])     # top-left:  effort vs rear safety
    ax2 = fig.add_subplot(gs[0, 1])     # top-right: effort vs forward safety
    ax3 = fig.add_subplot(gs[1, :])     # bottom (spans both cols): relative metric bars

    # Panel 1: effort vs min h_r (centroid + 1-sigma ellipse). Legend = A/B/C/D only;
    # the reference lines are labelled inline (left edge) to keep the legend minimal.
    for grp in ORDER:
        if grp not in groups:
            continue
        s = groups[grp]
        mean = (s['effort']['mean'], s['min_h_r']['mean'])
        _ellipse(ax1, mean, s['cov_effort_hr'], GROUP_COLORS[grp])
        ax1.plot(*mean, 'o', color=GROUP_COLORS[grp], ms=7, mec='k', zorder=5, label=grp)
    ax1.axhline(0.0, color='red', ls='--', lw=1.5)
    ax1.axhline(d_min, color='gray', ls='--', lw=1.3)
    ax1.text(0.008, 0.0, f'collision ({HR}=0)', transform=ax1.get_yaxis_transform(),
             ha='left', va='bottom', color='red', fontsize=INL_FS)
    ax1.text(0.008, d_min, DMIN_LBL, transform=ax1.get_yaxis_transform(),
             ha='left', va='bottom', color='dimgray', fontsize=INL_FS)
    ax1.set_xlabel(f'control intervention  {INT}   (lower = less conservative)', fontsize=LBL_FS)
    ax1.set_ylabel(f'min rear gap {HR} [m]  (higher = safer)', fontsize=LBL_FS)
    ax1.set_title(f'Effort vs rear safety (centroid + {SIGMA})', fontsize=TTL_FS)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=LEG_FS, loc='lower right', markerscale=1.6)

    # Panel 2: effort vs min h_f
    for grp in ORDER:
        if grp not in groups:
            continue
        s = groups[grp]
        mean = (s['effort']['mean'], s['min_h_f']['mean'])
        _ellipse(ax2, mean, s['cov_effort_hf'], GROUP_COLORS[grp])
        ax2.plot(*mean, 'o', color=GROUP_COLORS[grp], ms=7, mec='k', zorder=5, label=grp)
    ax2.axhline(0.0, color='orange', ls='--', lw=1.5)
    ax2.text(0.008, 0.0, f'lead collision ({HF}=0)', transform=ax2.get_yaxis_transform(),
             ha='left', va='bottom', color='darkorange', fontsize=INL_FS)
    ax2.set_xlabel(f'control intervention  {INT}', fontsize=LBL_FS)
    ax2.set_ylabel(f'min forward gap {HF} [m]', fontsize=LBL_FS)
    ax2.set_title(f'Effort vs forward safety (centroid + {SIGMA})', fontsize=TTL_FS)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=LEG_FS, loc='lower right', markerscale=1.6)

    # Panel 3 (spans bottom row): per-group metrics RELATIVE to the largest group per metric,
    # with the relative value printed on each bar. A is largest here -> A=1.00 reference.
    metrics = [('effort', f'effort  {INT}', 'effort'),
               ('min_h_f', f'min forward gap {HF}', HF),
               ('min_h_r', f'min rear gap {HR}', HR)]
    slot = 0.19          # center-to-center spacing of the 4 group bars (< 1 metric slot)
    bw = 0.155           # bar width (< slot -> small visible gap between bars)
    refs = []
    for mi, (m, _long, short) in enumerate(metrics):
        means = {g: groups[g][m]['mean'] for g in ORDER if g in groups}
        ref_g = max(means, key=means.get)                 # the "largest group" for this metric
        ref = means[ref_g]
        refs.append(f"{short}: {ref_g}={ref:.1f}")
        for gi, grp in enumerate(ORDER):
            if grp not in groups:
                continue
            rel = groups[grp][m]['mean'] / ref if ref else 0.0
            relstd = groups[grp][m]['std'] / abs(ref) if ref else 0.0
            xpos = mi + (gi - 1.5) * slot
            ax3.bar(xpos, rel, bw, yerr=relstd, capsize=2, color=GROUP_COLORS[grp],
                    alpha=0.85, label=grp if mi == 0 else None)
            ax3.text(xpos, rel + relstd + 0.015, f'{rel:.2f}', ha='center', va='bottom',
                     fontsize=NUM_FS)
    ax3.axhline(1.0, color='gray', ls=':', lw=1.0)
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels([_long for _, _long, _ in metrics], fontsize=LBL_FS)
    ax3.set_ylabel('relative to largest group (=1.0)', fontsize=LBL_FS)
    ax3.set_ylim(0, 1.32)
    ax3.set_title('Per-group metrics, each normalized to the largest group   ('
                  + ',   '.join(refs) + ')', fontsize=TTL_FS)
    ax3.legend(fontsize=LEG_FS, ncol=4, loc='upper center')
    ax3.grid(alpha=0.3, axis='y')

    # ---- color-coded group descriptions as an extended title (top of the figure) ----
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS[g],
                      markeredgecolor='k', markersize=13) for g in ORDER if g in groups]
    labels = [f"{groups[g]['label']} {DASH} rear-safe {groups[g]['pct_collision_free']:.0f}{PCT}"
              for g in ORDER if g in groups]
    leg = fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.965),
                     fontsize=HDR_FS, frameon=True, columnspacing=4.0, handletextpad=0.6,
                     borderpad=0.8, labelspacing=0.6)
    for txt, g in zip(leg.get_texts(), [g for g in ORDER if g in groups]):
        txt.set_color(GROUP_COLORS[g]); txt.set_fontweight('bold')

    meta = agg.get('meta', {})
    fig.suptitle(f"Sandwich bcbf interaction-handling stress test "
                 f"(n={meta.get('n_draws', '?')} draws, {PM}{int(100*meta.get('frac', 0))}{PCT} rear, "
                 f"seed {meta.get('seed', '?')})", fontsize=SUP_FS, y=0.99)
    # Save directly (not via save_figure) so its tight_layout doesn't override the manual
    # top spacing reserved for the color-coded header legend.
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {out_png}")


def print_summary(agg):
    print(f"\n{'group':>26} {'n':>3} {'effort(mean)':>12} {'min_h_r(mean)':>13} "
          f"{'safe%':>6} {'clrDmin%':>9} {'fwdSafe%':>9}")
    for grp in ORDER:
        s = agg['groups'].get(grp)
        if not s:
            continue
        print(f"{s['label']:>26} {s['n']:>3} {s['effort']['mean']:>12.2f} "
              f"{s['min_h_r']['mean']:>13.2f} {s['pct_collision_free']:>6.0f} "
              f"{s['pct_clears_dmin']:>9.0f} {s['pct_front_safe']:>9.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir',
                    default=os.path.join(_HERE, 'output', 'sandwich_bcbf_stress'))
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frac', type=float, default=0.40)
    ap.add_argument('--pilot', action='store_true',
                    help='deterministic scale sweep (separation check) instead of the MC')
    ap.add_argument('--from-raw', default=None, help='skip sims: re-aggregate + plot from a raw csv')
    ap.add_argument('--from-aggregated', default=None, help='skip to plot from an aggregated json')
    ap.add_argument('--no-latex', action='store_true',
                    help='disable the LaTeX/Times paper styling (default on); use default fonts')
    ap.add_argument('--jobs', type=int, default=1,
                    help='parallel worker processes for the MC (default 1 = sequential)')
    # scenario / ISSf tuning knobs (no code edits needed)
    ap.add_argument('--gap-r0', type=float, default=4.5)
    ap.add_argument('--lead-a-brake', type=float, default=1.5)
    ap.add_argument('--backup-beta-f', type=float, default=1.0)
    ap.add_argument('--rear-kappa', type=float, default=0.7)
    ap.add_argument('--l-inter-ratio', type=float, default=4.0)
    ap.add_argument('--l-inter-residue', type=float, default=0.5)
    ap.add_argument('--rho-rear', type=float, default=3e4)
    ap.add_argument('--gamma', type=float, default=2.0)
    args = ap.parse_args()

    out = args.output_dir
    raw_path = os.path.join(out, 'raw', 'stress_raw.csv')
    agg_path = os.path.join(out, 'aggregated', 'stress_aggregated.json')
    png_path = os.path.join(out, 'stress_clusters.png')

    scn = {'gap_r0': args.gap_r0, 'lead_a_brake': args.lead_a_brake,
           'backup_beta_f': args.backup_beta_f, 'rear_kappa': args.rear_kappa}
    issf = {'l_inter_ratio': args.l_inter_ratio, 'l_inter_residue': args.l_inter_residue,
            'rho_rear': args.rho_rear, 'gamma': args.gamma}
    meta = {'n_draws': args.n, 'seed': args.seed, 'frac': args.frac,
            'nominal': NOM, 'scenario': scn, 'issf': issf}

    # --- plot only -------------------------------------------------------------
    if args.from_aggregated:
        with open(args.from_aggregated) as fh:
            agg = json.load(fh)
        print_summary(agg)
        plot_aggregated(agg, png_path, use_tex=not args.no_latex)
        return

    # --- (re)aggregate + plot from existing raw -------------------------------
    if args.from_raw:
        rows = read_raw(args.from_raw)
        agg = aggregate(rows, meta)
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        with open(agg_path, 'w') as fh:
            json.dump(agg, fh, indent=2)
        print(f"Wrote aggregated -> {agg_path}")
        print_summary(agg)
        plot_aggregated(agg, png_path, use_tex=not args.no_latex)
        return

    # --- pilot: deterministic separation check (raw csv only) -----------------
    if args.pilot:
        rows = run_pilot(scn, issf, scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5])
        write_raw(rows, os.path.join(out, 'raw', 'stress_pilot.csv'))
        return

    # --- full MC: raw -> aggregated -> figure ---------------------------------
    rows = run_mc(scn, issf, args.n, args.seed, args.frac, jobs=args.jobs)
    write_raw(rows, raw_path)
    agg = aggregate(rows, meta)
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, 'w') as fh:
        json.dump(agg, fh, indent=2)
    print(f"Wrote aggregated -> {agg_path}")
    print_summary(agg)
    plot_aggregated(agg, png_path, use_tex=not args.no_latex)


if __name__ == '__main__':
    main()
