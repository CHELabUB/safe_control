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
# 'F' is the forward-CBF-only baseline (no rear awareness) -- special-cased everywhere.
GROUPS = {
    'A': ('passive', False),
    'B': ('nominal', False),
    'C': ('nominal', True),
    'D': ('oracle',  True),
}
FWD = 'F'                                                    # forward-only baseline key
GROUP_LABELS = {
    'F': 'F: forward CBF only (no rear)',
    'A': 'A: worst-case (passive)',
    'B': 'B: nominal, no buffer',
    'C': 'C: nominal + ISSf buffer',
    'D': 'D: oracle + buffer',
}
# Colours consistent with the motivating story figure: F forward-only = red, A worst-case = green,
# B no-buffer = blue, C nominal+buffer = purple, D oracle = grey.
GROUP_COLORS = {'F': 'tab:red', 'A': 'tab:green', 'B': 'tab:blue', 'C': 'tab:purple', 'D': 'tab:gray'}
ORDER = ['A', 'B', 'C', 'D']                                 # rear-aware groups (set the scatter y-limit)
ORDER_ALL = ['F', 'A', 'B', 'C', 'D']                        # incl. the forward-only baseline


def build_argv(group, a, b, scn, issf):
    """Construct the run_sandwich CLI argv for one (group, actual rear) combination."""
    if group == FWD:                                          # forward CBF only: no rear awareness
        return ['--method', 'forward_only',
                '--rear-kappa', str(scn['rear_kappa']), '--gap-r0', str(scn['gap_r0']),
                '--lead-a-brake', str(scn['lead_a_brake']),
                '--a-lead-decel', str(scn['lead_a_brake']),
                '--rear-alpha', str(a), '--rear-beta', str(b)]
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
    method = 'forward_only' if group == FWD else 'sandwiched_bcbf'
    args = RS.build_parser().parse_args(build_argv(group, a, b, scn, issf))
    cfg = RS.build_cfg(args, method)
    dt = args.dt
    n_sim = int(round(args.tf / dt))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        out = RS.simulate_sandwich(cfg, dt, n_sim, method)
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


def run_groups_over_draws(groups_list, draws, scn, issf, jobs=1):
    """Run the given groups over the given (alpha,beta) draws; return raw rows (Pool if jobs>1)."""
    tasks = [(di, g, a, b) for di, (a, b) in enumerate(draws) for g in groups_list]
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
    else:
        for k, task in enumerate(tasks, 1):
            r = _mc_task(task, scn, issf)
            rows.append(r)
            di, _, a, b = task
            print(f"  [{k:3d}/{total}] draw {di:2d} (a={a:.3f},b={b:.3f}) {r['group']}: "
                  f"min_h_f={r['min_h_f']:6.2f} min_h_r={r['min_h_r']:6.2f} "
                  f"effort={r['effort']:6.2f} buf={r['rear_buffer']:.2f}", flush=True)
    rows.sort(key=lambda r: (int(r['draw']), groups_list.index(r['group'])))
    return rows


def run_mc(scn, issf, n, seed, frac, jobs=1):
    """Run the paired MC over the rear-aware groups (A-D); return list of raw rows."""
    return run_groups_over_draws(ORDER, draw_actuals(n, seed, frac), scn, issf, jobs=jobs)


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


# --- forward-only ('F') baseline: ISSf-independent, so computed once per draw set + cached ----

def draws_from_rows(rows, ref_group='A'):
    """Recover the (draw -> (alpha,beta)) pairing from an existing raw, using one group's rows.
    Lets us extend/assemble a run on EXACTLY its draws without re-deriving from seed/frac."""
    ref = sorted((r for r in rows if r['group'] == ref_group), key=lambda r: int(r['draw']))
    return [(float(r['alpha']), float(r['beta'])) for r in ref]


def forward_cache_path(draws, scn, cache_dir):
    """Cache file keyed by the draw tuples + the scenario knobs the forward ego actually sees."""
    import hashlib
    key = repr([(round(a, 6), round(b, 6)) for a, b in draws]) + \
        repr((scn['gap_r0'], scn['lead_a_brake'], scn['rear_kappa']))
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f'forward_n{len(draws)}_{h}.csv')


def forward_rows(draws, scn, jobs=1, cache_path=None):
    """Forward-only ('F') rows over the given draws; load from cache_path if present, else compute."""
    if cache_path and os.path.exists(cache_path):
        rows = [r for r in read_raw(cache_path) if r['group'] == FWD]
        print(f"  forward-only: loaded {len(rows)} rows from cache {cache_path}")
        return rows
    tasks = [(di, FWD, a, b) for di, (a, b) in enumerate(draws)]
    issf = {}                                                # unused by forward-only
    if jobs and jobs > 1:
        from multiprocessing import Pool
        from functools import partial
        with Pool(processes=jobs) as pool:
            rows = list(pool.imap_unordered(partial(_mc_task, scn=scn, issf=issf), tasks))
        rows.sort(key=lambda r: int(r['draw']))
    else:
        rows = [_mc_task(t, scn, issf) for t in tasks]
    if cache_path:
        write_raw(rows, cache_path)
        print(f"  forward-only: computed + cached {len(rows)} rows -> {cache_path}")
    return rows


def aggregate(rows, meta):
    """Per-group summary (means/stds + covariances for ellipses + safety rates)."""
    d_min = float(rows[0]['d_min']) if 'd_min' in rows[0] else meta.get('d_min', 1.0)
    groups = {}
    for group in ORDER_ALL:
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
    S = 1.2 * 1.3                                        # ~1.3x larger panel fonts (paper use)
    LEG_FS, NUM_FS, LBL_FS, TTL_FS = 15 * S, 13 * S, 13 * S, 15 * S
    INL_FS = 13 * S                                       # inline reference-line / annotation labels
    HDR_FS, INFO_FS = 18, 14                              # header group key / run-info (fixed, fit width)
    BRK = dict(color='0.35', lw=1.3, ls=(0, (5, 4)), clip_on=False, zorder=10)   # axis-break dashes

    # Mode-aware label fragments: LaTeX/Times (tex) vs plain unicode (else).
    INT = r'$\int\!\left|u-u_{\mathrm{nom}}\right|\,\mathrm{d}t$' if tex else '∫|u-u_nom| dt'
    HF = r'$h_{\rm f}$' if tex else 'h_f'
    HR = r'$h_{\rm r}$' if tex else 'h_r'
    SIGMA = r'$1\sigma$' if tex else '1σ'
    PM = r'$\pm$' if tex else '±'
    PCT = r'\%' if tex else '%'
    DASH = '--' if tex else '—'
    DMIN_LBL = (r'$d_{\min}=%.1f$' % d_min) if tex else f'd_min={d_min}'

    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.27, wspace=0.11,
                          top=0.78, bottom=0.06, left=0.07, right=0.97)
    # Bottom (spans both cols): relative metric bars on a BROKEN y-axis -- a tall positive
    # segment (ax3t) and a short, compressed negative segment (ax3b) for the forward-only
    # rear-end bar, with a break symbol so the negative scale is clearly not the same.
    gs_bar = gs[1, :].subgridspec(2, 1, height_ratios=[4.0, 1.0], hspace=0.05)
    ax3t = fig.add_subplot(gs_bar[0])
    ax3b = fig.add_subplot(gs_bar[1], sharex=ax3t)

    # The two scatter panels use a 2x2 BROKEN x- and y-axis: the A-D cluster fills the zoomed
    # top-right cell while the forward-only outlier (low effort, far-off gap) sits in its own
    # bottom-left cell. Break marks on the interior edges flag the discontinuities.
    present = [g for g in ORDER_ALL if g in groups]
    clusterg = [g for g in ORDER if g in groups]

    def _sd(grp, cov_key, i):
        return float(np.sqrt(max(np.array(groups[grp][cov_key])[i, i], 0.0)))

    # Shared effort (x) windows: narrow left segment for forward-only, zoomed right for A-D.
    eff = {g: groups[g]['effort']['mean'] for g in present}
    esd = {g: _sd(g, 'cov_effort_hr', 0) for g in present}
    ce_lo = min(eff[g] - esd[g] for g in clusterg)
    ce_hi = max(eff[g] + esd[g] for g in clusterg)
    er = max(ce_hi - ce_lo, 1e-6)
    x_right = (ce_lo - 0.12 * er, ce_hi + 0.10 * er)
    x_left = (eff[FWD] - 0.10 * er, eff[FWD] + 0.16 * er)

    def broken_panel(cell, stat, cov_key, title, xlab, show_dmin, zero_in_top, coll_label):
        sg = cell.subgridspec(2, 2, width_ratios=[1.0, 3.6], height_ratios=[3.6, 1.0],
                              wspace=0.05, hspace=0.08)
        tl = fig.add_subplot(sg[0, 0]); tr = fig.add_subplot(sg[0, 1], sharey=tl)
        bl = fig.add_subplot(sg[1, 0], sharex=tl)
        br = fig.add_subplot(sg[1, 1], sharex=tr, sharey=bl)
        chm = [groups[g][stat]['mean'] for g in clusterg]
        chs = [_sd(g, cov_key, 1) for g in clusterg]
        clo = min(m - s for m, s in zip(chm, chs)); chi = max(m + s for m, s in zip(chm, chs))
        cr = max(chi - clo, 1e-6)
        top_lo = (min(clo, 0.0) if zero_in_top else clo) - 0.10 * cr
        top_hi = chi + 0.16 * cr
        fH = groups[FWD][stat]['mean']
        # Bottom cell: tight around F (and the 0 line when it lives down here), with ABSOLUTE
        # margins so a wide cluster range never inflates / overlaps it.
        if zero_in_top:                 # bottom cell holds only F (collision line is up top)
            half = max(0.5, 0.12 * cr)
            bot_lo, bot_hi = fH - half, fH + half
        else:                           # bottom cell holds F together with the collision line
            lo, hi = min(fH, 0.0), max(fH, 0.0)
            m = max(0.35, 0.4 * (hi - lo))
            bot_lo, bot_hi = lo - m, hi + m
        tl.set_ylim(top_lo, top_hi); bl.set_ylim(bot_lo, bot_hi)
        tl.set_xlim(*x_left); tr.set_xlim(*x_right)

        for grp in clusterg:            # A-D ellipses + centroids in the zoomed top-right cell
            m = (groups[grp]['effort']['mean'], groups[grp][stat]['mean'])
            _ellipse(tr, m, groups[grp][cov_key], GROUP_COLORS[grp])
            tr.plot(*m, 'o', color=GROUP_COLORS[grp], ms=9, mec='k', zorder=5)
        bl.plot(eff[FWD], fH, marker='v', color=GROUP_COLORS[FWD], ms=16, mec='k', zorder=6)

        for ax in (tl, tr, bl, br):     # lines/shading on every cell; each shows in-window only
            ax.axhline(0.0, color='red', ls='--', lw=1.5)
            if show_dmin:
                ax.axhline(d_min, color='gray', ls='--', lw=1.3)
            ax.axhspan(ax.get_ylim()[0], 0.0, color='red', alpha=0.08, zorder=0)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=LEG_FS)
        if show_dmin:
            tr.text(0.99, d_min, DMIN_LBL, transform=tr.get_yaxis_transform(),
                    ha='right', va='top', color='dimgray', fontsize=INL_FS)
        cax = tr if zero_in_top else br
        cax.text(0.99, 0.0, coll_label, transform=cax.get_yaxis_transform(),
                 ha='right', va='top', color='red', fontsize=INL_FS)

        # hide interior spines + duplicate tick labels
        tl.spines['right'].set_visible(False); tl.spines['bottom'].set_visible(False)
        tr.spines['left'].set_visible(False); tr.spines['bottom'].set_visible(False)
        bl.spines['right'].set_visible(False); bl.spines['top'].set_visible(False)
        br.spines['left'].set_visible(False); br.spines['top'].set_visible(False)
        tl.tick_params(labelbottom=False, bottom=False)
        tr.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        br.tick_params(labelleft=False, left=False)
        # break = two parallel dashed lines at each interior edge (the hspace/wspace gap separates
        # the pair). Horizontal pair for the row break; vertical pair for the column break.
        for ax in (tl, tr):                       # row break: bottom edge of the top row
            ax.plot([0, 1], [0, 0], transform=ax.transAxes, **BRK)
        for ax in (bl, br):                       # ... and top edge of the bottom row
            ax.plot([0, 1], [1, 1], transform=ax.transAxes, **BRK)
        for ax in (tl, bl):                       # column break: right edge of the left column
            ax.plot([1, 1], [0, 1], transform=ax.transAxes, **BRK)
        for ax in (tr, br):                       # ... and left edge of the right column
            ax.plot([0, 0], [0, 1], transform=ax.transAxes, **BRK)
        # compact [m] unit label between the top two ticks of the left cell
        ticks = [t for t in tl.get_yticks() if top_lo <= t <= top_hi]
        yv = 0.5 * (ticks[-1] + ticks[-2]) if len(ticks) >= 2 else top_lo + 0.8 * cr
        tl.text(-0.04, yv, '[m]', transform=tl.get_yaxis_transform(),
                ha='right', va='center', fontsize=LBL_FS)
        # title + xlabel centered over the whole cell (figure coords)
        bb = cell.get_position(fig)
        fig.text(bb.x0 + bb.width / 2, bb.y1 + 0.004, title, ha='center', va='bottom', fontsize=TTL_FS)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.048, xlab, ha='center', va='top', fontsize=LBL_FS)

    broken_panel(gs[0, 0], 'min_h_r', 'cov_effort_hr', f'Effort vs rear safety (min {HR})',
                 f'control intervention  {INT}  (lower = cheaper)', show_dmin=True,
                 zero_in_top=True, coll_label=f'collision ({HR}=0)')
    broken_panel(gs[0, 1], 'min_h_f', 'cov_effort_hf', f'Effort vs forward safety (min {HF})',
                 f'control intervention  {INT}', show_dmin=False,
                 zero_in_top=False, coll_label=f'lead collision ({HF}=0)')

    # Bottom (broken y-axis): per-group metrics RELATIVE to the largest group per metric, with the
    # ACTUAL value +/- 1 sigma printed on each bar. Positive bars live on ax3t; the forward-only
    # rear-end bar (negative) lives on the short, compressed ax3b.
    metrics = [('effort', f'effort  {INT}', 'effort'),
               ('min_h_f', f'min forward gap {HF}', HF),
               ('min_h_r', f'min rear gap {HR}', HR)]
    bar_order = [g for g in ORDER_ALL if g in groups]      # F first, then A-D
    nb = len(bar_order)
    slot = 0.16          # center-to-center spacing of the group bars (< 1 metric slot)
    bw = 0.14            # bar width (< slot -> small visible gap between bars)
    bar_tops, bar_bots = [], [0.0]                         # rel+-std per bar, for auto y-limits
    for mi, (m, _long, short) in enumerate(metrics):
        # Reference = largest among the rear-aware groups (A here); F is small/negative so it
        # never sets the reference. Forward-only's h_r is negative -> its bar points downward.
        means = {g: groups[g][m]['mean'] for g in ORDER if g in groups}
        ref = means[max(means, key=means.get)]
        for gi, grp in enumerate(bar_order):
            rel = groups[grp][m]['mean'] / ref if ref else 0.0
            relstd = groups[grp][m]['std'] / abs(ref) if ref else 0.0
            xpos = mi + (gi - (nb - 1) / 2.0) * slot
            for ax in (ax3t, ax3b):                        # draw on both; each window clips
                ax.bar(xpos, rel, bw, yerr=relstd, capsize=2, color=GROUP_COLORS[grp], alpha=0.85)
            gmean, gstd = groups[grp][m]['mean'], groups[grp][m]['std']
            lbl = f'{gmean:.1f}\n{PM}{gstd:.1f}'
            if rel >= 0:
                ax3t.text(xpos, rel + relstd + 0.02, lbl, ha='center', va='bottom',
                          fontsize=NUM_FS, linespacing=0.9)
                bar_tops.append(rel + relstd)
            else:
                # negative bar (forward-only h_r): print the value ABOVE the bar -- just over the
                # 0 baseline in the top segment -- rather than below the downward bar.
                ax3t.text(xpos, 0.05, lbl, ha='center', va='bottom',
                          fontsize=NUM_FS, linespacing=0.9)
                bar_bots.append(rel - relstd)
    ax3t.axhline(1.0, color='gray', ls=':', lw=1.0)
    ax3t.set_ylim(0.0, max(1.40, max(bar_tops) + 0.24))           # positive segment
    ax3b.set_ylim(min(bar_bots) - 0.15, 0.0)                      # compressed negative segment
    for ax in (ax3t, ax3b):
        ax.grid(alpha=0.3, axis='y')
    # Break the axis between the two segments: two parallel dashed lines (the hspace gap separates
    # the bottom edge of the top segment from the top edge of the bottom segment).
    ax3t.spines['bottom'].set_visible(False)
    ax3b.spines['top'].set_visible(False)
    ax3t.tick_params(labelbottom=False, bottom=False)
    ax3t.plot([0, 1], [0, 0], transform=ax3t.transAxes, **BRK)
    ax3b.plot([0, 1], [1, 1], transform=ax3b.transAxes, **BRK)
    ax3b.set_xticks(range(len(metrics)))
    ax3b.set_xticklabels([_long for _, _long, _ in metrics], fontsize=LBL_FS)
    ax3t.set_ylabel('relative to largest group (=1.0)', fontsize=LBL_FS)
    ax3t.yaxis.set_label_coords(-0.055, 0.35)                     # center label across both segments

    # ---- color-coded group descriptions as an extended title (top of the figure) ----
    present = [g for g in ORDER_ALL if g in groups]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS[g],
                      markeredgecolor='k', markersize=13) for g in present]
    labels = [f"{groups[g]['label']} {DASH} rear-safe {groups[g]['pct_collision_free']:.0f}{PCT}"
              for g in present]
    # All run information lives in the top header block (no separate suptitle):
    # the color-coded group key, plus the MC / scenario / ISSf parameters as its title.
    meta = agg.get('meta', {})
    scn = meta.get('scenario', {})
    issf = meta.get('issf', {})
    nom = meta.get('nominal', list(NOM))
    AB = r'$\alpha/\beta$' if tex else 'α/β'
    LIN = r'$L_{\rm inter}$' if tex else 'L_inter'
    RHOR = r'$\rho_{\rm rear}$' if tex else 'rho_rear'
    GAM = r'$\gamma$' if tex else 'gamma'
    KAP = r'$\kappa$' if tex else 'kappa'
    BETAF = r'$\beta_{\rm f}$' if tex else 'beta_f'
    GAPR = r'gap$_{r0}$' if tex else 'gap_r0'
    info1 = (f"{meta.get('n_draws', '?')} draws, {PM}{int(100 * meta.get('frac', 0))}{PCT} rear "
             f"spread, seed {meta.get('seed', '?')}    |    nominal rear {AB}="
             f"{nom[0]:g}/{nom[1]:g}    |    {DMIN_LBL}")
    info2 = (f"scenario: {GAPR}={scn.get('gap_r0', 0):g} m, lead brake={scn.get('lead_a_brake', 0):g}, "
             f"backup {BETAF}={scn.get('backup_beta_f', 0):g}, rear {KAP}={scn.get('rear_kappa', 0):g}"
             f"     ISSf (C,D): {LIN}={issf.get('l_inter_ratio', 0):g}, "
             f"residue={issf.get('l_inter_residue', 0):g}, {RHOR}={issf.get('rho_rear', 0):g}, "
             f"{GAM}={issf.get('gamma', 0):g}")
    leg = fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.975),
                     fontsize=HDR_FS, frameon=True, columnspacing=4.0, handletextpad=0.6,
                     borderpad=0.9, labelspacing=0.6, title=info1 + '\n' + info2,
                     labelcolor=[GROUP_COLORS[g] for g in present])   # color by label order (robust)
    leg.get_title().set_fontsize(INFO_FS)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')
    # Save directly (not via save_figure) so its tight_layout doesn't override the manual
    # top spacing reserved for the color-coded header legend.
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {out_png}")


def print_summary(agg):
    print(f"\n{'group':>30} {'n':>3} {'effort(mean)':>12} {'min_h_r(mean)':>13} "
          f"{'safe%':>6} {'clrDmin%':>9} {'fwdSafe%':>9}")
    for grp in ORDER_ALL:
        s = agg['groups'].get(grp)
        if not s:
            continue
        print(f"{s['label']:>30} {s['n']:>3} {s['effort']['mean']:>12.2f} "
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
    # forward-only ('F') baseline + group reuse
    ap.add_argument('--with-forward', action='store_true',
                    help="include the forward-CBF-only baseline 'F' (computed once per draw set, cached)")
    ap.add_argument('--forward-cache', default=None,
                    help='dir for the forward-only cache (default output/_forward_cache)')
    ap.add_argument('--reuse-from', default=None,
                    help='assemble a new variant: take --reuse-groups rows from this raw csv, '
                         'compute only the remaining rear-aware groups, on the SAME draws')
    ap.add_argument('--reuse-groups', default='A,B',
                    help='comma-separated groups to reuse from --reuse-from (default A,B)')
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
    fwd_dir = args.forward_cache or os.path.join(_HERE, 'output', '_forward_cache')

    def add_forward(rows, draws=None):
        """Append the forward-only 'F' rows (load from cache or compute once for the draw set)."""
        draws = draws if draws is not None else draws_from_rows(rows)
        cpath = forward_cache_path(draws, scn, fwd_dir)
        frows = forward_rows(draws, scn, jobs=args.jobs, cache_path=cpath)
        return rows + frows, os.path.relpath(cpath, _HERE)

    def finish(rows, provenance):
        write_raw(rows, raw_path)
        m = dict(meta); m.update(provenance)
        agg = aggregate(rows, m)
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        with open(agg_path, 'w') as fh:
            json.dump(agg, fh, indent=2)
        print(f"Wrote aggregated -> {agg_path}")
        print_summary(agg)
        plot_aggregated(agg, png_path, use_tex=not args.no_latex)

    # --- plot only -------------------------------------------------------------
    if args.from_aggregated:
        with open(args.from_aggregated) as fh:
            agg = json.load(fh)
        print_summary(agg)
        plot_aggregated(agg, png_path, use_tex=not args.no_latex)
        return

    # --- assemble a NEW variant: reuse A/B from a base raw, compute only C/D ----
    if args.reuse_from:
        base = read_raw(args.reuse_from)
        reuse = [g.strip() for g in args.reuse_groups.split(',') if g.strip()]
        draws = draws_from_rows(base)
        reused_rows = [r for r in base if r['group'] in reuse]
        compute_groups = [g for g in ORDER if g not in reuse]
        print(f"Reuse {reuse} from {args.reuse_from}; compute {compute_groups} for new ISSf "
              f"({len(draws)} draws)")
        rows = reused_rows + run_groups_over_draws(compute_groups, draws, scn, issf, jobs=args.jobs)
        prov = {'reuse_from': os.path.relpath(args.reuse_from, _HERE),
                'groups_reused': reuse, 'groups_computed': compute_groups}
        if args.with_forward:
            rows, cpath = add_forward(rows, draws)
            prov['groups_computed'] = compute_groups + [FWD]
            prov['forward_only_cache'] = cpath
        finish(rows, prov)
        return

    # --- (re)aggregate + plot from existing raw (optionally inject forward-only) -
    if args.from_raw:
        rows = read_raw(args.from_raw)
        # Preserve the source run's meta (n/seed/frac/issf/scenario) so the header is accurate,
        # rather than the CLI defaults. Look for the sibling aggregated json next to the raw.
        src_agg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.from_raw))),
                               'aggregated', 'stress_aggregated.json')
        if os.path.exists(src_agg):
            src_meta = json.load(open(src_agg)).get('meta', {})
            meta.update({k: src_meta[k] for k in ('n_draws', 'seed', 'frac', 'issf', 'nominal')
                         if k in src_meta})
            scn.update(src_meta.get('scenario', {}))
        prov = {}
        if args.with_forward:
            rows, cpath = add_forward(rows)
            prov = {'assembled': 'forward-only injected into existing A-D raw',
                    'forward_only_cache': cpath}
        finish(rows, prov)
        return

    # --- pilot: deterministic separation check (raw csv only) -----------------
    if args.pilot:
        rows = run_pilot(scn, issf, scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5])
        write_raw(rows, os.path.join(out, 'raw', 'stress_pilot.csv'))
        return

    # --- full MC: raw -> aggregated -> figure ---------------------------------
    rows = run_mc(scn, issf, args.n, args.seed, args.frac, jobs=args.jobs)
    prov = {}
    if args.with_forward:
        rows, cpath = add_forward(rows, draw_actuals(args.n, args.seed, args.frac))
        prov['forward_only_cache'] = cpath
    finish(rows, prov)


if __name__ == '__main__':
    main()
