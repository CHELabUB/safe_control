"""
Animation maker for the rear-aware sandwich runs.

Renders a saved ``series_*.npz`` (lead -> ego -> rear, 1-D longitudinal) as a top-down
single-lane traffic movie: a black lead car, a coloured ego car, and a yellow rear car
driving along the gray drift-track road, with a rich HUD (time, speeds, h_f/h_r, backup
indicator) and a red collision flash when a gap goes negative.

Geometry (per the rear-aware convention, ego is the camera reference):
    X_ego(k) = s_ego[k] - s_ego[0]                      # ego starts at 0, camera follows it
    lead center = X_ego + L + h_f + BUFFER              # neighbours reconstructed from the
    rear center = X_ego - L - h_r - BUFFER              #   gaps, not from s_lead / s_rear
with L = BODY_LENGTH and a BUFFER = 5 m baked into every shown gap so that collisions
(h < 0) stay visually separated instead of fully overlapping.

Ego colour: default = orange car. If specified it maps to the stress/story group colour:
    F / forward_only -> red,  A / worst_case -> green,  B / nominal_no_buffer -> blue,
    C / nominal_issf -> purple,  D / oracle -> grey.
Lead is always black, rear always yellow.

Two entry paths:
    # single run (folder or direct .npz); colour optional
    uv run python examples/rear_aware/make_video.py <run_dir_or_npz> [--color F]

    # batch via a JSON spec ({"output_dir": ..., "runs": {"label": {"path":, "color":}}})
    uv run python examples/rear_aware/make_video.py --spec videos/sandwich_story/story_videos.json

Outputs an ``.mp4`` (ffmpeg) next to the run, falling back to ``.gif`` if ffmpeg is absent.
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from rear_aware_common import plt, BODY_LENGTH, load_run_series, _HERE   # noqa: E402
from plot_runs import find_series, resolve_run                          # noqa: E402
from safe_control.envs.drifting_env import DriftingEnv                  # noqa: E402

L = BODY_LENGTH
BUFFER = 3.0                       # render-only gap buffer (m)
ELEMENTS = os.path.join(_HERE, 'videos', 'elements')

# --- colour resolution -----------------------------------------------------------------
VALID_CARS = {'black', 'orange', 'yellow', 'red', 'green', 'blue', 'purple', 'grey'}
COLOR_ALIASES = {
    'f': 'red', 'forward': 'red', 'forward_only': 'red',
    'a': 'green', 'worst': 'green', 'worst_case': 'green', 'passive': 'green',
    'b': 'blue', 'no_buffer': 'blue', 'nominal_no_buffer': 'blue',
    'c': 'purple', 'issf': 'purple', 'nominal_issf': 'purple',
    'd': 'grey', 'gray': 'grey', 'oracle': 'grey',
}


def _set_paper_style():
    """Times + LaTeX rendering (matches the figure plots; requires latex + newtx)."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',
        'mathtext.fontset': 'stix',      # Times-like fallback for the fast (mathtext) HUD
        'axes.unicode_minus': False,
    })


def _set_plain_style():
    """Default matplotlib fonts (no LaTeX); STIX (Times-like) for the mathtext HUD."""
    plt.rcParams.update({'text.usetex': False, 'font.family': 'sans-serif',
                         'mathtext.fontset': 'stix', 'axes.unicode_minus': True})


def resolve_color(spec):
    """Map a CLI/spec colour token (group key, alias, or raw colour) to a car PNG name."""
    if not spec:
        return 'orange'
    s = str(spec).strip().lower()
    if s in VALID_CARS:
        return s
    if s in COLOR_ALIASES:
        return COLOR_ALIASES[s]
    raise SystemExit(f"unknown --color '{spec}'; use a group key (F/A/B/C/D), an alias "
                     f"(worst_case/nominal_issf/...), or a car colour {sorted(VALID_CARS)}")


def load_sprite(color):
    """Load a car PNG as RGBA; white-key sprites whose background is opaque white."""
    img = mpimg.imread(os.path.join(ELEMENTS, f'top_view_{color}_car.png')).astype(float)
    if img.shape[-1] == 3:                                   # add an alpha channel
        img = np.dstack([img, np.ones(img.shape[:2])])
    # Some assets (black, purple) ship with a fully-opaque white background -> key it out.
    corners = np.concatenate([img[:8, :8, 3].ravel(), img[:8, -8:, 3].ravel(),
                              img[-8:, :8, 3].ravel(), img[-8:, -8:, 3].ravel()])
    if corners.mean() > 0.9:
        rgb = img[..., :3]
        white = np.all(rgb > 0.92, axis=-1)
        img[..., 3][white] = 0.0
    return img


# --- series -> displayed positions -----------------------------------------------------
def _pad(a, n):
    """Pad a 1-D array to length n by repeating its last sample (h_* are len = n-1)."""
    a = np.asarray(a, float)
    if len(a) >= n:
        return a[:n]
    return np.concatenate([a, np.full(n - len(a), a[-1])])


def displayed_positions(series, buffer=BUFFER):
    """Reconstruct (t, x_lead, x_ego, x_rear, h_f, h_r) in the render frame.

    Ego keeps its true travel (x_ego = s_ego - s_ego[0]); the neighbours are placed from
    the *gaps* with a small render buffer baked in so collisions stay legible.
    """
    t = series['t_state']
    n = len(t)
    x_ego = series['s_ego'] - series['s_ego'][0]
    hf = _pad(series['h_f'], n)
    hr = _pad(series['h_r'], n)
    x_lead = x_ego + L + hf + buffer
    x_rear = x_ego - L - hr - buffer
    return t, x_lead, x_ego, x_rear, hf, hr


# --- rendering -------------------------------------------------------------------------
def render_video(series, color, out_path, label='', fps=20, dpi=130,
                 window=None, hud=True, step=1, buffer=BUFFER):
    """Render one run to a video file (mp4, gif fallback).

    The camera tracks a point moving at the run's *average* speed (constant velocity), so
    the cars visibly speed up / fall back relative to it. Two profile panels (speeds and
    headways) are stacked under the lane, each with a circle marker pinned at the current
    sample.
    """
    t, x_lead, x_ego, x_rear, hf, hr = displayed_positions(series, buffer)
    n = len(t)
    v_lead = series.get('v_lead'); v_ego = series['v_ego']; v_rear = series['v_rear']
    backup = series.get('backup_active')
    S = 1.4                                         # global font-size scale

    # constant-velocity camera: centre = platoon centroid at t0, advancing at mean speed
    v_avg = float(np.mean(v_ego))
    centroid0 = (x_lead[0] + x_ego[0] + x_rear[0]) / 3.0
    xcam = centroid0 + v_avg * (t - t[0])
    stack = np.stack([x_lead, x_ego, x_rear])
    excursion = float(np.max(np.abs(stack - xcam[None, :])))
    halfwin = (window / 2.0) if window else (excursion + 0.6 * (L + buffer) + 2.0)

    # --- figure: lane on top, speed + headway profiles below -----------------------------
    env = DriftingEnv(track_type='straight', track_width=7.0, track_length=10.0, num_lanes=1)
    yspan = 6.5
    fig_w = 13.0
    margin_l, margin_r = 0.07, 0.985
    road_h = fig_w * (margin_r - margin_l) * (2 * yspan) / (2 * halfwin)
    plot_h = 2.4
    fig_h = road_h + plot_h + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 2, height_ratios=[road_h, plot_h], hspace=0.30, wspace=0.16)
    fig.subplots_adjust(left=margin_l, right=margin_r, top=0.99, bottom=0.11)
    ax = fig.add_subplot(gs[0, :])          # road spans both columns
    ax_v = fig.add_subplot(gs[1, 0])        # speeds  (narrow, left)
    ax_h = fig.add_subplot(gs[1, 1])        # headways (narrow, right)

    # road strip spanning every camera window over the whole run
    road_lo = float((xcam - halfwin).min()) - 6.0
    road_hi = float((xcam + halfwin).max()) + 6.0
    env.centerline = np.column_stack([np.linspace(road_lo, road_hi, 300), np.zeros(300)])
    hw = env.track_width / 2.0
    env.left_boundary = np.column_stack([env.centerline[:, 0], np.full(300, hw)])
    env.right_boundary = np.column_stack([env.centerline[:, 0], np.full(300, -hw)])
    env.center_line_color = 'white'                                # white centre line
    env.setup_plot(ax=ax)
    ax.set_xlabel(''); ax.set_ylabel(''); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(-yspan, yspan)
    ax.set_aspect('equal')
    ax.set_xlim(xcam[0] - halfwin, xcam[0] + halfwin)

    # roadside markers every 10 m (fixed world coords) -> a scrolling motion reference;
    # white, no outline, sitting on the road boundary so they read as boundary markers
    for xm in np.arange(np.floor(road_lo / 10.0) * 10.0, road_hi + 10.0, 10.0):
        for yb in (hw, -hw):
            ax.add_patch(Rectangle((xm - 0.6, yb - 0.4), 1.2, 0.8, facecolor='white',
                                   edgecolor='none', zorder=3))

    # car sprites (all face +x; no flip needed)
    sprites = {'lead': load_sprite('black'), 'ego': load_sprite(color),
               'rear': load_sprite('yellow')}
    fig.canvas.draw()                                              # fix transData scale
    ppx = (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
    boxes, outlines = {}, {}
    for key in ('lead', 'ego', 'rear'):
        spr = sprites[key]
        zoom = L * ppx / spr.shape[1]                              # span ~L metres in x
        ab = AnnotationBbox(OffsetImage(spr, zoom=zoom), (0, 0), frameon=False, zorder=5)
        ax.add_artist(ab)
        boxes[key] = ab
        cw = spr.shape[0] / spr.shape[1] * L                       # car width (m)
        rect = Rectangle((0, 0), L + 1.0, cw + 0.8, fill=False, edgecolor='red',
                         lw=3.0, zorder=6, visible=False)
        ax.add_patch(rect)
        outlines[key] = (rect, L + 1.0, cw + 0.8)

    # HUD uses mathtext (STIX, Times-like) not usetex -> fast per-frame redraw
    hud_txt = ax.text(0.012, 0.96, '', transform=ax.transAxes, va='top', ha='left',
                      fontsize=13 * S, zorder=10,
                      bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.85))
    hud_txt.set_usetex(False)
    status_txt = ax.text(0.5, 0.92, '', transform=ax.transAxes, va='top', ha='center',
                         fontsize=19 * S, fontweight='bold', color='red', zorder=10)
    if label:
        ax.text(0.988, 0.96, label, transform=ax.transAxes, va='top', ha='right',
                fontsize=14 * S, zorder=10,
                bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.85))

    # --- static profile curves + live markers --------------------------------------------
    cmap = {'lead': 'black', 'ego': color, 'rear': 'gold'}
    ax_v.plot(t, v_lead if v_lead is not None else np.full(n, np.nan),
              color=cmap['lead'], lw=1.8)
    ax_v.plot(t, v_ego, color=cmap['ego'], lw=1.8)
    ax_v.plot(t, v_rear, color=cmap['rear'], lw=1.8)
    ax_v.set_title('speed [m/s]', fontsize=14 * S)
    ax_v.set_xlabel('time [s]', fontsize=12 * S)
    ax_v.set_xlim(t[0], t[-1]); ax_v.grid(alpha=0.3)
    ax_v.tick_params(labelsize=11 * S)
    mk_v = {key: ax_v.plot([], [], 'o', color=cmap[key], ms=9, mec='k', mew=0.8,
                           zorder=5)[0] for key in ('lead', 'ego', 'rear')}

    # h_f follows the ego car colour, h_r the rear (yellow) car colour
    h_lo = min(float(np.min(hf)), float(np.min(hr)), -1.0) - 0.6
    h_hi = max(float(np.max(hf)), float(np.max(hr))) + 0.6
    ax_h.set_ylim(h_lo, h_hi)
    ax_h.axhspan(h_lo, 0.0, color='red', alpha=0.12, zorder=0)      # collision region
    ax_h.axhline(0.0, color='red', ls='--', lw=1.3, zorder=1)
    ax_h.plot(t, hf, color=cmap['ego'], lw=1.8, zorder=3)
    ax_h.plot(t, hr, color='gold', lw=1.8, zorder=3)
    ax_h.set_title('headway [m]', fontsize=14 * S)
    ax_h.set_xlabel('time [s]', fontsize=12 * S)
    ax_h.set_xlim(t[0], t[-1]); ax_h.grid(alpha=0.3)
    ax_h.tick_params(labelsize=11 * S)
    mk_hf, = ax_h.plot([], [], 'o', color=cmap['ego'], ms=10, mec='k', mew=0.8, zorder=5)
    mk_hr, = ax_h.plot([], [], 'o', color='gold', ms=10, mec='k', mew=0.8, zorder=5)

    def place(rect_tuple, x):
        rect, w, h = rect_tuple
        rect.set_xy((x - w / 2.0, -h / 2.0))

    def update(k):
        boxes['lead'].xybox = boxes['lead'].xy = (x_lead[k], 0.0)
        boxes['ego'].xybox = boxes['ego'].xy = (x_ego[k], 0.0)
        boxes['rear'].xybox = boxes['rear'].xy = (x_rear[k], 0.0)
        ax.set_xlim(xcam[k] - halfwin, xcam[k] + halfwin)
        rear_hit, lead_hit = hr[k] < 0, hf[k] < 0
        for key, (rx, hit) in (('lead', (x_lead[k], lead_hit)),
                               ('ego', (x_ego[k], rear_hit or lead_hit)),
                               ('rear', (x_rear[k], rear_hit))):
            rt = outlines[key]
            place(rt, rx)
            rt[0].set_visible(hit)
        # live markers on the profiles
        if v_lead is not None:
            mk_v['lead'].set_data([t[k]], [v_lead[k]])
        mk_v['ego'].set_data([t[k]], [v_ego[k]])
        mk_v['rear'].set_data([t[k]], [v_rear[k]])
        mk_hf.set_data([t[k]], [hf[k]])
        mk_hr.set_data([t[k]], [hr[k]])
        if hud:
            # the profile panels already show the speeds/headways -> HUD keeps only the
            # clock and the backup-status line
            lines = [rf"$t = {t[k]:.2f}\ [\mathrm{{s}}]$"]
            if backup is not None and bool(backup[min(k, len(backup) - 1)]):
                lines.append(r"$\mathrm{BACKUP\ ACTIVE}$")
            hud_txt.set_text("\n".join(lines))
        msg = []
        if rear_hit:
            msg.append("REAR-END")
        if lead_hit:
            msg.append("LEAD COLLISION")
        status_txt.set_text("  ".join(msg))
        return ()

    frames = range(0, n, max(1, step))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps, blit=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    base, _ = os.path.splitext(out_path)
    try:
        anim.save(base + '.mp4', writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=dpi)
        saved = base + '.mp4'
    except Exception as e:                                          # ffmpeg unavailable
        print(f"  ffmpeg save failed ({e}); falling back to gif")
        anim.save(base + '.gif', writer=PillowWriter(fps=fps), dpi=max(80, dpi // 2))
        saved = base + '.gif'
    plt.close(fig)
    print(f"Saved animation -> {saved}")
    return saved


# --- entry points ----------------------------------------------------------------------
def _slug(s):
    out = ''.join(c if c.isalnum() else '_' for c in s).lower()
    while '__' in out:                                  # collapse runs of separators
        out = out.replace('__', '_')
    return out.strip('_')


def run_single(run, color, out, **kw):
    bases = [os.getcwd(), _HERE]
    if run.endswith('.npz'):
        npz = run
    else:
        folder = run if os.path.isdir(run) else resolve_run(run, bases)
        npz = find_series(folder)
    series = load_run_series(npz)
    method = os.path.basename(npz)[len('series_'):-len('.npz')]
    if out is None:
        out = os.path.join(os.path.dirname(npz), f'video_{method}.mp4')
    render_video(series, resolve_color(color), out, label=kw.pop('label', ''), **kw)


def run_spec(spec_path, out_dir_override=None, **kw):
    with open(spec_path) as fh:
        spec = json.load(fh)
    spec_dir = os.path.dirname(os.path.abspath(spec_path))
    bases = [_HERE, spec_dir, os.getcwd()]
    runs = spec['runs'] if 'runs' in spec else spec
    out_dir = out_dir_override or spec.get('output_dir') or spec_dir
    if not os.path.isabs(out_dir):
        for b in bases:
            cand = os.path.normpath(os.path.join(b, out_dir))
            if os.path.isdir(cand) or b == spec_dir:
                out_dir = cand
                break
    for label, entry in runs.items():
        rel = entry['path'] if isinstance(entry, dict) else entry
        color = entry.get('color') if isinstance(entry, dict) else None
        folder = resolve_run(rel, bases)
        npz = find_series(folder)
        series = load_run_series(npz)
        out = os.path.join(out_dir, f'video_{_slug(label)}.mp4')
        print(f"[{label}] {os.path.relpath(npz)} -> color={resolve_color(color)}")
        render_video(series, resolve_color(color), out, label=label, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?', help='run folder or series_*.npz (single-run mode)')
    ap.add_argument('--spec', help='JSON spec for batch rendering (one video per run)')
    ap.add_argument('--color', help='ego car colour / group key (default: orange)')
    ap.add_argument('--out', help='output path (single-run mode)')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--dpi', type=int, default=130)
    ap.add_argument('--window', type=float, default=None, help='camera width in metres')
    ap.add_argument('--buffer', type=float, default=BUFFER,
                    help='render-only gap buffer in metres (default: %(default)s)')
    ap.add_argument('--step', type=int, default=1, help='frame decimation')
    ap.add_argument('--no-hud', action='store_true')
    ap.add_argument('--no-latex', action='store_true',
                    help='disable LaTeX/Times rendering (use if latex is unavailable)')
    args = ap.parse_args()

    _set_plain_style() if args.no_latex else _set_paper_style()
    kw = dict(fps=args.fps, dpi=args.dpi, window=args.window, hud=not args.no_hud,
              step=args.step, buffer=args.buffer)
    if args.spec:
        run_spec(args.spec, **kw)
    elif args.run:
        run_single(args.run, args.color, args.out, **kw)
    else:
        ap.error('provide a run path or --spec')


if __name__ == '__main__':
    main()
