"""
Backup CBF demonstration for the 1D double integrator.

System:  x = [s, v],  x_dot = [v, u]
Safety:  h(x) = s >= 0
Input:   u in [u_min, u_max]
Nominal: u_nom = u_min  (drives s toward 0)
Backup:  u_b (constant positive acceleration, typically u_max)

Analytical invariant sets
--------------------------
  S_max  = maximal forward invariant set (under optimal control u_max)
         = { s > 0                     if v >= 0
           { s - v^2/(2*u_max) >= 0    if v < 0
  Derivation: worst-case recovery with u = u_max; min s = s - v^2/(2*u_max).

  S(T)   = T-horizon backup-safe set (under constant u_b for time T)
         = { s > 0                  if v >= 0 }  AND  v >= -u_b*T
           { s - v^2/(2*u_b) > 0   if v < 0  }
  Derivation: backup traj min s = s - v^2/(2*u_b); reachability needs v+u_b*T > 0.

  As T -> inf,  S(T) -> S_max  (when u_b = u_max).

Usage:
    uv run python examples/double_integrator/run_backup_cbf_1d.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Local imports (Python adds script directory to sys.path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from double_integrator_1d import DoubleIntegrator1D
from backup_cbf_1d_wrapper import BackupCBF1D

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DT = 0.02
T_SIM = 4.0           # total simulation duration (seconds)
BACKUP_HORIZON = 2.0  # backup horizon T
U_NOM = -1.0          # nominal control (tries to violate s >= 0)
U_B = 1.0             # backup control
U_MAX = 1.0
EPS_S, EPS_V = 0.05, 0.05

N_SIM = int(T_SIM / DT)

ROBOT_SPEC = {'u_max': U_MAX, 'a_max': U_MAX}

ALPHA_CONFIGS = {
    'linear':     lambda h: h,
    'cubic':      lambda h: h + h ** 3,
    'aggressive': lambda h: 5 * h + h ** 3,
}

# Initial conditions for trajectory showcase
# Format: (s0, v0, label)
SHOWCASE_ICS = [
    (0.8,  -1.0, 'in S(T), slow'),
    (0.5,  -0.5, 'in S(T), fast'),
    (0.2,  -0.5, 'near boundary'),
    (0.8,  -2.5, 'outside S(T), v<-T'),
]

# -----------------------------------------------------------------------
# Analytical set membership
# -----------------------------------------------------------------------

def in_S_max(s: float, v: float, u_max: float = U_MAX) -> bool:
    """True if (s,v) is in the maximal forward invariant set under optimal control u_max."""
    if v >= 0:
        return s > 0
    return (s - v ** 2 / (2 * u_max)) > 0


def in_S_T(s: float, v: float, T: float, u_b: float = U_B) -> bool:
    """True if (s,v) is in the T-horizon backup-safe set S(T) under backup control u_b."""
    s_min = (s - v ** 2 / (2 * u_b)) if v < 0 else s
    return s_min > 0 and v >= -u_b * T


# -----------------------------------------------------------------------
# Simulation helpers
# -----------------------------------------------------------------------

def make_cbf(alpha_fn, alpha_terminal_fn=None):
    robot = DoubleIntegrator1D(DT, dict(ROBOT_SPEC))
    cbf = BackupCBF1D(
        robot=robot,
        robot_spec=dict(ROBOT_SPEC),
        dt=DT,
        backup_horizon=BACKUP_HORIZON,
        u_b=U_B,
        alpha_fn=alpha_fn,
        alpha_terminal_fn=alpha_terminal_fn,
        eps_s=EPS_S,
        eps_v=EPS_V,
    )
    cbf.set_nominal_controller(lambda x: np.array([U_NOM]))
    return robot, cbf


def run_sim(s0: float, v0: float, alpha_fn, alpha_terminal_fn=None):
    """Run closed-loop simulation; returns (states, controls, safe_flag)."""
    robot, cbf = make_cbf(alpha_fn, alpha_terminal_fn)
    x = np.array([s0, v0])
    states = [x.copy()]
    controls = []

    for _ in range(N_SIM):
        u_safe = cbf.solve_control_problem(x)
        u = float(u_safe.flat[0])
        x = np.array([x[0] + x[1] * DT, x[1] + u * DT])
        states.append(x.copy())
        controls.append(u)

    states = np.array(states)       # (N_SIM+1, 2)
    controls = np.array(controls)   # (N_SIM,)
    h_traj = states[:, 0]           # h(x) = s
    safe_flag = bool(np.all(h_traj >= -1e-4))
    return states, controls, h_traj, safe_flag


# -----------------------------------------------------------------------
# Figure 1: Phase portrait + verification
# -----------------------------------------------------------------------

def plot_phase_portrait(ax):
    """Draw S_max and S(T) regions + sampled verification dots + trajectories."""

    T = BACKUP_HORIZON

    # --- Shade S_max and S(T) regions ---
    s_grid = np.linspace(0, 2.5, 400)
    v_grid = np.linspace(-3.2, 2.5, 400)
    SS, VV = np.meshgrid(s_grid, v_grid)

    mask_smax = np.zeros_like(SS, dtype=bool)
    mask_st = np.zeros_like(SS, dtype=bool)
    for i in range(len(v_grid)):
        for j in range(len(s_grid)):
            s, v = SS[i, j], VV[i, j]
            mask_smax[i, j] = in_S_max(s, v, U_MAX)
            mask_st[i, j] = in_S_T(s, v, T, U_B)

    ax.contourf(SS, VV, mask_smax.astype(float), levels=[0.5, 1.5],
                colors=['#d4edda'], alpha=0.5, zorder=1)
    ax.contourf(SS, VV, mask_st.astype(float), levels=[0.5, 1.5],
                colors=['#90ee90'], alpha=0.6, zorder=2)

    # S_max boundary: parabola s = v^2/(2*u_max) for v < 0
    v_neg = np.linspace(-3.2, 0, 200)
    s_boundary = v_neg ** 2 / (2 * U_MAX)
    ax.plot(s_boundary, v_neg, 'k-', lw=2.0, label=r'$S_{\max}$ boundary', zorder=5)

    # s = 0 boundary (vertical)
    ax.axvline(0, color='k', lw=1.5, ls='--', zorder=4)

    # S(T) extra boundary: v = -u_b*T horizontal line
    v_st_bound = -U_B * T
    ax.axhline(v_st_bound, color='navy', lw=1.5, ls='--',
               label=fr'$v = -u_b T = {v_st_bound:.1f}$ (S(T) boundary)', zorder=5)

    # --- Sampled verification points ---
    # Sample on a coarse grid and run short simulations
    s_samples = np.linspace(0.05, 2.0, 8)
    v_samples = np.linspace(-3.0, 1.5, 8)
    alpha_fn = ALPHA_CONFIGS['linear']

    for s0 in s_samples:
        for v0 in v_samples:
            in_set = in_S_T(s0, v0, T, U_B)
            states, _, h_traj, safe = run_sim(s0, v0, alpha_fn)

            if in_set and safe:
                color, marker, zorder = '#2ca02c', 'o', 8
            elif in_set and not safe:
                color, marker, zorder = 'orange', 'X', 10   # CBF failure
            elif not in_set and not safe:
                color, marker, zorder = '#d62728', 's', 7
            else:
                color, marker, zorder = '#ff7f0e', '^', 9   # outside S(T) but safe

            ax.scatter(s0, v0, c=color, marker=marker, s=40,
                       zorder=zorder, edgecolors='none', alpha=0.85)

    # --- Showcase trajectories ---
    colors_traj = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
    for (s0, v0, label), color in zip(SHOWCASE_ICS, colors_traj):
        states, _, _, _ = run_sim(s0, v0, alpha_fn)
        ax.plot(states[:, 0], states[:, 1], '-', color=color, lw=1.8,
                label=label, zorder=12, alpha=0.9)
        ax.plot(s0, v0, 'D', color=color, ms=7, zorder=13,
                markeredgecolor='k', markeredgewidth=0.5)

    # --- Legend for scatter ---
    legend_patches = [
        mpatches.Patch(facecolor='#90ee90', edgecolor='none', alpha=0.8, label=r'$S(T)$ region'),
        mpatches.Patch(facecolor='#d4edda', edgecolor='none', alpha=0.7, label=r'$S_{\max} \setminus S(T)$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', ms=7,
               label='in S(T), safe ✓'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', ms=7,
               label='outside S(T), unsafe ✓'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', ms=8,
               label='in S(T), violated! ✗'),
    ]

    ax.legend(handles=legend_patches + [
        Line2D([0], [0], c='k', lw=2, label=r'$S_{\max}$ boundary'),
        Line2D([0], [0], c='navy', lw=1.5, ls='--', label=fr'$v=-u_b T$ boundary'),
    ], loc='upper right', fontsize=7, framealpha=0.9)

    ax.set_xlabel('s (position)', fontsize=11)
    ax.set_ylabel('v (velocity)', fontsize=11)
    ax.set_title(f'Phase portrait  (T={BACKUP_HORIZON}s, α=linear, u_nom={U_NOM})', fontsize=11)
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-3.3, 2.5)
    ax.set_aspect('equal')
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(True, lw=0.4, alpha=0.4)


# -----------------------------------------------------------------------
# Figure 2: h(x(t)) time series
# -----------------------------------------------------------------------

def plot_h_timeseries(ax):
    """Plot h(x(t)) = s(t) vs time for showcase trajectories and alpha configs."""
    t_arr = np.linspace(0, T_SIM, N_SIM + 1)
    colors_traj = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
    ls_map = {'linear': '-', 'cubic': '--', 'aggressive': ':'}

    for (s0, v0, label), color in zip(SHOWCASE_ICS[:2], colors_traj[:2]):
        for alpha_name, alpha_fn in ALPHA_CONFIGS.items():
            _, _, h_traj, _ = run_sim(s0, v0, alpha_fn)
            ax.plot(t_arr, h_traj, ls_map[alpha_name], color=color,
                    lw=1.5, alpha=0.85,
                    label=f'{label}, α={alpha_name}' if alpha_name == 'linear' else None)

    ax.axhline(0, color='k', lw=1.5, ls='-', label='h = 0 (constraint)')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('h(x) = s (position)', fontsize=11)
    ax.set_title('Safety CBF value h(x(t)) over time', fontsize=11)

    # Legend for line styles
    ls_legend = [
        Line2D([0], [0], c='gray', lw=1.5, ls='-', label='α: linear'),
        Line2D([0], [0], c='gray', lw=1.5, ls='--', label='α: cubic'),
        Line2D([0], [0], c='gray', lw=1.5, ls=':', label='α: aggressive'),
        Line2D([0], [0], c='#1f77b4', lw=2, label=SHOWCASE_ICS[0][2]),
        Line2D([0], [0], c='#ff7f0e', lw=2, label=SHOWCASE_ICS[1][2]),
        Line2D([0], [0], c='k', lw=1.5, ls='-', label='h = 0'),
    ]
    ax.legend(handles=ls_legend, fontsize=7, loc='upper right')
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_ylim(-0.2, None)


# -----------------------------------------------------------------------
# Figure 3: Analytical vs numerical backup trajectory
# -----------------------------------------------------------------------

def plot_analytical_vs_numerical(ax_state, ax_error):
    """
    Compare analytical and numerical backup trajectories from a test point.

    For this nilpotent linear system (A^2 = 0):
      - S_numerical = S_analytical exactly (Euler STM is exact)
      - phi_numerical has O(dt) per-step position error vs phi_analytical
    """
    x_test = np.array([0.6, -1.0])   # (s, v) — inside S(T)
    T = BACKUP_HORIZON

    robot, cbf = make_cbf(ALPHA_CONFIGS['linear'])
    cmp = cbf.compare_trajectories(x_test)

    t_arr = np.arange(cbf.N) * DT

    phi_a = cmp['phi_analytical']
    phi_n = cmp['phi_numerical']

    ax_state.plot(t_arr, phi_a[:, 0], 'b-', lw=2.0, label='Analytical s(t)')
    ax_state.plot(t_arr, phi_n[:, 0], 'r--', lw=1.5, label='Numerical s(t)')
    ax_state.plot(t_arr, phi_a[:, 1], 'b:', lw=1.5, label='Analytical v(t)')
    ax_state.plot(t_arr, phi_n[:, 1], 'r-.', lw=1.2, label='Numerical v(t)')
    ax_state.axhline(0, color='k', lw=1, ls='--', alpha=0.4)
    ax_state.set_xlabel('Backup time (s)', fontsize=10)
    ax_state.set_ylabel('State', fontsize=10)
    ax_state.set_title(
        rf'Backup traj from $x_0=({x_test[0]},{x_test[1]})$', fontsize=10
    )
    ax_state.legend(fontsize=7)
    ax_state.grid(True, lw=0.4, alpha=0.4)

    # Error plot
    ax_error.semilogy(t_arr, cmp['phi_error_norm'] + 1e-16, 'b-',
                      lw=1.8, label='||φ_a - φ_n||')
    ax_error.semilogy(t_arr, cmp['S_error_norm'] + 1e-16, 'r--',
                      lw=1.5, label='||S_a - S_n||_F')
    ax_error.set_xlabel('Backup time (s)', fontsize=10)
    ax_error.set_ylabel('Error (log scale)', fontsize=10)
    ax_error.set_title(
        f'Discretization error  (dt={DT})', fontsize=10
    )
    ax_error.legend(fontsize=8)
    ax_error.grid(True, lw=0.4, alpha=0.4)
    ax_error.text(
        0.97, 0.05,
        f'max φ err: {cmp["max_phi_error"]:.2e}\nmax S err: {cmp["max_S_error"]:.2e}',
        transform=ax_error.transAxes, ha='right', va='bottom', fontsize=7,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6)
    )


# -----------------------------------------------------------------------
# Figure 4: Alpha function shapes
# -----------------------------------------------------------------------

def plot_alpha_shapes(ax):
    """Visualize the class-K alpha functions."""
    h_vals = np.linspace(-0.5, 2.0, 300)
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    for (name, fn), color in zip(ALPHA_CONFIGS.items(), colors):
        ax.plot(h_vals, [fn(h) for h in h_vals], color=color, lw=2.0, label=rf'α: {name}')

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('h', fontsize=11)
    ax.set_ylabel('α(h)', fontsize=11)
    ax.set_title('Class-K alpha functions', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print(f"Running Backup CBF 1D demo  (T={BACKUP_HORIZON}s, dt={DT}s, u_nom={U_NOM})")
    print(f"Sampling grid for verification (8×8 = 64 points) ...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.38)

    ax_phase = fig.add_subplot(gs[:, 0:2])  # left half: phase portrait (tall)
    ax_h = fig.add_subplot(gs[0, 2])
    ax_state = fig.add_subplot(gs[1, 2])

    # Extra small figure for error and alpha shapes
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    fig2.subplots_adjust(wspace=0.35)

    print("  Plotting phase portrait ...")
    plot_phase_portrait(ax_phase)

    print("  Plotting h timeseries ...")
    plot_h_timeseries(ax_h)

    print("  Comparing analytical vs numerical trajectories ...")
    plot_analytical_vs_numerical(ax_state, axes2[0])

    plot_alpha_shapes(axes2[1])

    fig.suptitle(
        'Backup CBF: 1D double integrator  '
        r'($\dot{s}=v$, $\dot{v}=u$, $s\geq 0$)',
        fontsize=13, y=0.98
    )

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, 'backup_cbf_1d.png'), dpi=120, bbox_inches='tight')
    fig2.savefig(os.path.join(fig_dir, 'backup_cbf_1d_analysis.png'), dpi=120, bbox_inches='tight')
    print(f"Saved figures to {fig_dir}/")
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')


if __name__ == '__main__':
    main()
