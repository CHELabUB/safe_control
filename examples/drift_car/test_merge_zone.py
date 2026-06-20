"""
Created on June 20th, 2026
@author: Chaozhe He

@description:
Merge zone test case for safety shielding algorithms (Gatekeeper, MPS, BackupCBF).

Three-vehicle scenario on a 3-lane straight track:
  - Ego vehicle:   upper lane (lane 0), initial x=0, travels at target_velocity
  - Stalled car:   upper lane (lane 0), x = stalled_x_offset ahead of ego (static)
  - Moving car:    middle lane (lane 1), x = moving_x_offset relative to ego,
                   travels at target_velocity (same speed, ignores ego)
  - Lower lane (lane 2): always empty, serves as backup escape route

Nominal plan (MPCC): S-curve reference path upper→middle (around stalled car)→upper
Backup policy: LaneChangeController to lower lane

Key insight:
  Gatekeeper validates the full [nominal + backup] trajectory and detects early that
  the S-curve nominal path will conflict with the moving car in the middle lane.
  It switches to the lower-lane backup while ego is still safely in the upper lane.

  MPS and BackupCBF have less lookahead: they may follow the nominal step-by-step
  until ego is committed to the merge, at which point the backup path through
  the moving car's position becomes unsafe.

Tunable parameters (all relative to ego at x=0):
  --stalled   x-offset of stalled car in upper lane (default 50m)
  --moving    x-offset of moving car in middle lane (default 5m)

Usage:
    uv run python examples/drift_car/test_merge_zone.py --algo gatekeeper --stalled 50 --moving 5
    uv run python examples/drift_car/test_merge_zone.py --algo mps --stalled 50 --moving 5
    uv run python examples/drift_car/test_merge_zone.py --algo backupcbf --stalled 50 --moving 5
    uv run python examples/drift_car/test_merge_zone.py --sweep
    uv run python examples/drift_car/test_merge_zone.py --algo gatekeeper --save

@required-scripts: safe_control/shielding/gatekeeper.py, safe_control/shielding/mps.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
import re
from safe_control.envs.drifting_env import DriftingEnv
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator
from safe_control.position_control.mpcc import MPCC
from safe_control.position_control.backup_controller import LaneChangeController
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.utils.animation import AnimationSaver


# =============================================================================
# Algorithm Types
# =============================================================================

ALGO_TYPES = ['gatekeeper', 'mps', 'backupcbf']


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TrackConfig:
    """3-lane track: upper (0), middle (1), lower (2)."""
    track_type: str = 'straight'
    track_length: float = 200.0
    lane_width: float = 4.0
    num_lanes: int = 3
    ego_lane_idx: int = 0       # upper lane — ego starts here, stalled car here
    middle_lane_idx: int = 1    # moving vehicle travels here
    backup_lane_idx: int = 2    # lower lane — empty, backup escape route


@dataclass
class VehicleConfig:
    """Vehicle configuration — same dynamics as test_drift.py."""
    # Geometry
    a: float = 1.4
    b: float = 1.4
    wheel_base: float = 2.8
    body_length: float = 4.5
    body_width: float = 2.0
    radius: float = 1.2

    # Mass and inertia
    m: float = 2500.0
    Iz: float = 5000.0

    # Tire parameters
    Cc_f: float = 80000.0
    Cc_r: float = 100000.0
    mu: float = 1.0
    r_w: float = 0.35
    gamma: float = 0.95

    # Input limits
    delta_max: float = np.deg2rad(20)
    delta_dot_max: float = np.deg2rad(25)
    tau_max: float = 4000.0
    tau_dot_max: float = 8000.0

    # State limits
    v_max: float = 20.0
    v_min: float = 0.0
    r_max: float = 2.0
    beta_max: float = np.deg2rad(45)

    # MPCC specific
    v_psi_max: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'a': self.a, 'b': self.b, 'wheel_base': self.wheel_base,
            'body_length': self.body_length, 'body_width': self.body_width,
            'radius': self.radius, 'm': self.m, 'Iz': self.Iz,
            'Cc_f': self.Cc_f, 'Cc_r': self.Cc_r, 'mu': self.mu,
            'r_w': self.r_w, 'gamma': self.gamma,
            'delta_max': self.delta_max, 'delta_dot_max': self.delta_dot_max,
            'tau_max': self.tau_max, 'tau_dot_max': self.tau_dot_max,
            'v_max': self.v_max, 'v_min': self.v_min,
            'r_max': self.r_max, 'beta_max': self.beta_max,
            'v_psi_max': self.v_psi_max,
        }


@dataclass
class SimulationConfig:
    dt: float = 0.05
    tf: float = 15.0
    nominal_horizon_time: float = 6.0
    backup_horizon_time: float = 3.0
    event_offset: float = 0.05
    safety_margin: float = 0.01
    initial_velocity: float = 10.0
    target_velocity: float = 10.0


@dataclass
class TestConfig:
    name: str
    description: str
    track: TrackConfig
    vehicle: VehicleConfig
    simulation: SimulationConfig
    stalled_x_offset: float = 50.0   # x of stalled car in upper lane (relative to ego)
    moving_x_offset: float = 80.0    # x of moving car in middle lane (relative to ego)
    moving_vx: float = -10.0         # vx of moving car (negative = oncoming, positive = same dir)
    algo_type: str = 'gatekeeper'
    save_animation: bool = False
    expected_collision: bool = False


# =============================================================================
# Path Construction
# =============================================================================

def build_nominal_path(
    track_config: TrackConfig,
    stalled_x: float,
    ego_lane_y: float,
    middle_lane_y: float,
    track_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an S-curve nominal reference path: upper lane → middle lane → upper lane.

    The path avoids the stalled car by routing through the middle lane.
    Uses cosine interpolation for smooth lane transitions.
    """
    # Transition extents
    merge_start = max(5.0, stalled_x - 30.0)
    merge_end = max(merge_start + 10.0, stalled_x - 8.0)
    return_start = stalled_x + 8.0
    return_end = stalled_x + 30.0

    n_points = 400
    x = np.linspace(0, track_length, n_points)
    y = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < merge_start:
            # Straight in upper lane
            y[i] = ego_lane_y
        elif xi < merge_end:
            # Smooth cosine transition: upper → middle
            t = (xi - merge_start) / (merge_end - merge_start)
            y[i] = ego_lane_y + (middle_lane_y - ego_lane_y) * 0.5 * (1 - np.cos(np.pi * t))
        elif xi < return_start:
            # Straight in middle lane (past stalled car)
            y[i] = middle_lane_y
        elif xi < return_end:
            # Smooth cosine transition: middle → upper
            t = (xi - return_start) / (return_end - return_start)
            y[i] = middle_lane_y + (ego_lane_y - middle_lane_y) * 0.5 * (1 - np.cos(np.pi * t))
        else:
            # Straight in upper lane
            y[i] = ego_lane_y

    return x, y


# =============================================================================
# Test Environment Setup
# =============================================================================

def setup_environment(config: TestConfig) -> Tuple[DriftingEnv, plt.Axes, plt.Figure]:
    track = config.track
    total_width = track.lane_width * track.num_lanes

    env = DriftingEnv(
        track_type=track.track_type,
        track_width=total_width,
        track_length=track.track_length,
        num_lanes=track.num_lanes,
    )

    plt.ion()
    ax, fig = env.setup_plot()
    algo_name = config.algo_type.upper()
    fig.canvas.manager.set_window_title(
        f'{algo_name} Merge Zone | stalled={config.stalled_x_offset:.0f}m, '
        f'moving={config.moving_x_offset:+.0f}m'
    )
    return env, ax, fig


def setup_vehicle(
    config: TestConfig, env: DriftingEnv, ax: plt.Axes
) -> Tuple[DriftingCar, np.ndarray, float, float]:
    ego_lane_y = env.get_lane_center(config.track.ego_lane_idx)
    backup_lane_y = env.get_lane_center(config.track.backup_lane_idx)

    X0 = np.array([
        1.0,
        ego_lane_y,
        np.deg2rad(0),
        0, 0,
        config.simulation.initial_velocity,
        0, 0,
    ])

    robot_spec = config.vehicle.to_dict()
    robot_spec['v_ref'] = config.simulation.target_velocity
    robot_spec['safety_margin'] = config.simulation.safety_margin
    car = DriftingCar(X0, robot_spec, config.simulation.dt, ax)

    return car, X0, ego_lane_y, backup_lane_y


def setup_controllers(
    config: TestConfig,
    car: DriftingCar,
    env: DriftingEnv,
    ego_lane_y: float,
    backup_lane_y: float,
    ax: plt.Axes,
) -> Tuple[MPCC, Union[Gatekeeper, MPS, BackupCBF]]:
    sim = config.simulation
    track = config.track

    middle_lane_y = env.get_lane_center(track.middle_lane_idx)

    # Build S-curve nominal reference path
    ref_x, ref_y = build_nominal_path(
        track, config.stalled_x_offset, ego_lane_y, middle_lane_y, track.track_length
    )

    nominal_horizon_steps = int(sim.nominal_horizon_time / sim.dt)
    mpcc = MPCC(car, car.robot_spec, horizon=nominal_horizon_steps)
    mpcc.set_reference_path(ref_x, ref_y)
    mpcc.set_cost_weights(
        Q_c=8.0,
        Q_l=1.0,
        Q_theta=2.0,
        Q_v=50.0,
        Q_r=400.0,
        v_ref=sim.target_velocity,
        R=np.array([100.0, 0.5, 0.1]),
    )
    mpcc.set_progress_rate(sim.target_velocity)

    # Backup: lane change to lower lane (backup_lane_y < ego_lane_y → 'right' w.r.t. driving dir)
    backup_controller = LaneChangeController(car.robot_spec, sim.dt, direction='right')
    print(f"  Backup: lane change to lower lane (y={backup_lane_y:.2f})")

    if config.algo_type == 'mps':
        shielding = MPS(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            event_offset=sim.event_offset,
            ax=ax,
            safety_margin=sim.safety_margin,
        )
        print("  Algorithm: MPS (one-step nominal horizon)")
    elif config.algo_type == 'backupcbf':
        shielding = BackupCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            ax=ax,
        )
        print("  Algorithm: BackupCBF (instantaneous QP)")
    else:
        shielding = Gatekeeper(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            event_offset=sim.event_offset,
            ax=ax,
            nominal_horizon=sim.nominal_horizon_time,
            safety_margin=sim.safety_margin,
        )
        print("  Algorithm: Gatekeeper (backward search)")

    shielding.set_backup_controller(backup_controller, target=backup_lane_y)
    shielding.set_environment(env)
    if env.dynamic_obstacles:
        shielding.set_moving_obstacles(lambda t=0.0: env.get_dynamic_obstacle_states(t))

    return mpcc, shielding


def setup_obstacles(config: TestConfig, env: DriftingEnv):
    """Add stalled car (static) and moving car (dynamic) to the environment."""
    obstacle_spec = {
        'body_length': 4.5,
        'body_width': 2.0,
        'a': 1.4,
        'b': 1.4,
        'radius': 1.0,
    }
    track = config.track

    # Stalled car: static obstacle in upper lane
    stalled_y = env.get_lane_center(track.ego_lane_idx)
    env.add_obstacle_car(
        x=config.stalled_x_offset,
        y=stalled_y,
        theta=0.0,
        robot_spec=obstacle_spec,
    )
    print(f"  Stalled car: x={config.stalled_x_offset:.1f}, y={stalled_y:.2f} (upper lane)")

    # Moving car: dynamic obstacle in middle lane at constant velocity
    moving_y = env.get_lane_center(track.middle_lane_idx)
    env.add_moving_obstacle_car(
        x=config.moving_x_offset,
        y=moving_y,
        theta=0.0,
        vx=config.moving_vx,
        vy=0.0,
        robot_spec=obstacle_spec,
    )
    print(f"  Moving car:  x={config.moving_x_offset:+.1f}, y={moving_y:.2f} (middle lane), "
          f"vx={config.moving_vx:.1f} m/s")


def setup_visualization(
    ax: plt.Axes,
    env: DriftingEnv,
    ego_lane_y: float,
    backup_lane_y: float,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> Tuple:
    # Nominal S-curve reference path
    ax.plot(ref_x, ref_y, 'g-', linewidth=1.5, alpha=0.4, label='Nominal path (MPCC)')
    # Backup target lane
    ax.axhline(y=backup_lane_y, color='orange', linewidth=1, alpha=0.3,
               linestyle=':', label='Backup lane (lower)')
    # Ego start lane
    ax.axhline(y=ego_lane_y, color='cyan', linewidth=1, alpha=0.2,
               linestyle=':', label='Upper lane')

    ref_horizon_line, = ax.plot([], [], 'y-', linewidth=3, alpha=0.9, label='MPCC horizon')
    mpc_pred_line, = ax.plot([], [], 'r--', linewidth=2, alpha=0.8, label='MPCC prediction')

    ax.legend(loc='upper right', fontsize=7)
    return ref_horizon_line, mpc_pred_line


# =============================================================================
# Simulation Loop
# =============================================================================

def run_simulation(
    config: TestConfig,
    car: DriftingCar,
    env: DriftingEnv,
    mpcc: MPCC,
    shielding: Union[Gatekeeper, MPS, BackupCBF],
    simulator: DriftingCarSimulator,
    ref_horizon_line,
    mpc_pred_line,
    ax: plt.Axes,
    fig: plt.Figure,
    animation_saver: Optional[AnimationSaver] = None,
) -> Dict[str, Any]:
    sim = config.simulation
    robot_spec = config.vehicle.to_dict()

    num_steps = int(sim.tf / sim.dt)
    window_size = (60, 20)

    nominal_steps = 0
    backup_steps = 0
    collision_occurred = False
    collision_step = None

    print(f"\nRunning simulation for {sim.tf}s ({num_steps} steps)...")

    for step in range(num_steps):
        state = car.get_state()
        pos = car.get_position()

        # Get MPCC nominal plan (S-curve path)
        try:
            mpcc.solve_control_problem(state)
            pred_states, pred_controls = mpcc.get_full_predictions()
            if pred_states is not None and pred_controls is not None:
                shielding.set_nominal_trajectory(pred_states, pred_controls)
        except Exception as e:
            print(f"MPCC error at step {step}: {e}")
            pred_states, pred_controls = None, None

        # Shielding validates and returns committed control
        U = shielding.solve_control_problem(state, friction=car.get_friction())

        # Track mode
        if shielding.is_using_backup():
            backup_steps += 1
        else:
            nominal_steps += 1

        # Step physics (also steps dynamic obstacles internally)
        result = simulator.step(U)

        # Update visualizations
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])

        pred_states_viz, _ = mpcc.get_predictions()
        if pred_states_viz is not None:
            mpc_pred_line.set_data(pred_states_viz[0, :], pred_states_viz[1, :])

        env.update_plot_frame(ax, pos, window_size=window_size)
        simulator.draw_plot(pause=0.001)

        if animation_saver is not None:
            animation_saver.save_frame(fig)

        # Status output every 50 steps
        if step % 50 == 0:
            V = car.get_velocity()
            status = shielding.get_status()
            mode = "BACKUP" if status['using_backup'] else "NOMINAL"
            h_min = status.get('h_min', 1.0)
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, "
                  f"V={V:5.2f} m/s, mode={mode:8s}, h_min={h_min:6.3f}")

        # Collision check
        if result['collision']:
            collision_occurred = True
            collision_step = step
            collision_type = getattr(simulator, 'collision_type', 'unknown')
            print(f"\n*** COLLISION ({collision_type}) at step {step} ***")
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            if animation_saver is not None:
                animation_saver.save_frame(fig, force=True)
            plt.pause(2.0)
            break

        # End if reached track end
        if pos[0] > env.track_length - 10:
            print("\nReached end of track!")
            break

    total_steps = nominal_steps + backup_steps
    return {
        'collision': collision_occurred,
        'collision_step': collision_step,
        'total_steps': total_steps,
        'nominal_steps': nominal_steps,
        'backup_steps': backup_steps,
        'nominal_ratio': nominal_steps / max(total_steps, 1),
        'backup_ratio': backup_steps / max(total_steps, 1),
        'global_min_h': shielding.get_status().get('global_min_h', 0.0),
    }


# =============================================================================
# Headless Data Collection & Figure Generation
# =============================================================================

def collect_trajectory_data(
    algo_type: str,
    stalled_x_offset: float = 50.0,
    moving_x_offset: float = 80.0,
    moving_vx: float = -10.0,
) -> Dict[str, Any]:
    """Run simulation headlessly and return full per-step trajectory data."""
    config = create_merge_zone_test(
        algo_type=algo_type,
        stalled_x_offset=stalled_x_offset,
        moving_x_offset=moving_x_offset,
        moving_vx=moving_vx,
    )

    track = config.track
    total_width = track.lane_width * track.num_lanes
    env = DriftingEnv(
        track_type=track.track_type,
        track_width=total_width,
        track_length=track.track_length,
        num_lanes=track.num_lanes,
    )

    plt.ioff()
    fig_h, ax_h = plt.subplots(1, 1, figsize=(20, 6))
    plt.close(fig_h)

    car, X0, ego_lane_y, backup_lane_y = setup_vehicle(config, env, ax_h)
    setup_obstacles(config, env)
    mpcc, shielding = setup_controllers(config, car, env, ego_lane_y, backup_lane_y, ax_h)
    simulator = DriftingCarSimulator(car, env, show_animation=False)

    sim = config.simulation
    num_steps = int(sim.tf / sim.dt)

    xs, ys, modes, h_mins, times = [], [], [], [], []
    collision = False

    print(f"  Collecting trajectory: {algo_type} ...", end='', flush=True)
    for step in range(num_steps):
        state = car.get_state()
        pos = car.get_position()

        try:
            mpcc.solve_control_problem(state)
            pred_states, pred_controls = mpcc.get_full_predictions()
            if pred_states is not None and pred_controls is not None:
                shielding.set_nominal_trajectory(pred_states, pred_controls)
        except Exception:
            pass

        U = shielding.solve_control_problem(state, friction=car.get_friction())

        status = shielding.get_status()
        xs.append(pos[0])
        ys.append(pos[1])
        modes.append('backup' if status['using_backup'] else 'nominal')
        h_mins.append(status.get('h_min', 1.0))
        times.append(step * sim.dt)

        result = simulator.step(U)
        if result['collision']:
            collision = True
            break
        if pos[0] > env.track_length - 10:
            break

    print(f" {'COLLISION' if collision else 'OK'} ({len(xs)} steps)")
    return {
        'algo': algo_type,
        'x': np.array(xs),
        'y': np.array(ys),
        'mode': modes,
        'h_min': np.array(h_mins),
        't': np.array(times),
        'collision': collision,
        'ego_lane_y': ego_lane_y,
        'backup_lane_y': backup_lane_y,
        'middle_lane_y': env.get_lane_center(track.middle_lane_idx),
        'stalled_x': stalled_x_offset,
        'moving_x': moving_x_offset,
        'moving_vx': moving_vx,
        'track_length': track.track_length,
    }


def generate_trajectory_figure(
    stalled_x_offset: float = 50.0,
    moving_x_offset: float = 80.0,
    moving_vx: float = -10.0,
    output_path: str = 'output/merge_zone_comparison.png',
) -> None:
    """
    Generate a trajectory overlay figure for all three algorithms on the same scenario,
    styled similarly to shielding/sample_fig.png.

    Panel (a): Top-down view — ego trajectories (3 colours) + lane lines + obstacles
    Panel (b): h_min safety value over time per algorithm
    Panel (c): Active mode (nominal / backup) over time per algorithm
    """
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    algos = ['backupcbf', 'mps', 'gatekeeper']
    colors = {
        'backupcbf': '#1f77b4',   # blue  — matches sample_fig Backup CBF colour
        'mps':       '#9467bd',   # purple
        'gatekeeper': '#2ca02c',  # green
    }
    labels = {
        'backupcbf': 'Ego Car: Backup CBF',
        'mps':       'Ego Car: MPS',
        'gatekeeper': 'Ego Car: Gatekeeper',
    }
    linestyles = {'backupcbf': '-', 'mps': '--', 'gatekeeper': '-.'}

    print(f"\nGenerating trajectory comparison figure ...")
    print(f"  Scenario: stalled_x={stalled_x_offset:.0f}m, "
          f"moving_x={moving_x_offset:+.0f}m, moving_vx={moving_vx:+.1f} m/s")

    data = {}
    for algo in algos:
        data[algo] = collect_trajectory_data(algo, stalled_x_offset, moving_x_offset, moving_vx)

    ref_data = data[algos[0]]
    ego_lane_y = ref_data['ego_lane_y']
    backup_lane_y = ref_data['backup_lane_y']
    middle_lane_y = ref_data['middle_lane_y']
    stalled_x = ref_data['stalled_x']
    moving_x_init = ref_data['moving_x']
    track_length = ref_data['track_length']

    # Build the reference S-curve path for context
    tc = TrackConfig()
    ref_px, ref_py = build_nominal_path(tc, stalled_x, ego_lane_y, middle_lane_y, track_length)

    plt.ioff()
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)
    ax_traj = fig.add_subplot(gs[0, :])   # full-width top: trajectory
    ax_h    = fig.add_subplot(gs[1, 0])   # bottom-left: h_min
    ax_mode = fig.add_subplot(gs[1, 1])   # bottom-right: mode timeline

    # --- Panel (a): Trajectories ---
    ax_traj.set_facecolor('#f5f5f5')

    # Lane boundaries and centrelines
    lane_ys = [ego_lane_y, middle_lane_y, backup_lane_y]
    lane_names = ['Upper lane', 'Middle lane', 'Lower lane (backup)']
    lane_colors = ['#cccccc', '#bbbbbb', '#aaaaaa']
    lw = ref_data['track_length']
    half_lw = 2.0  # half of lane_width=4

    for i, ly in enumerate(lane_ys):
        ax_traj.axhspan(ly - half_lw, ly + half_lw,
                        alpha=0.12 + 0.04 * i, color=lane_colors[i])
        ax_traj.axhline(ly, color='white', linewidth=1.0, alpha=0.6, linestyle='--')

    # Dashed lane dividers
    for boundary_y in [ego_lane_y - half_lw, middle_lane_y - half_lw, backup_lane_y - half_lw,
                        backup_lane_y + half_lw]:
        ax_traj.axhline(boundary_y, color='gray', linewidth=1.5, alpha=0.8, linestyle='-')

    # Reference S-curve path
    ax_traj.plot(ref_px, ref_py, color='green', linewidth=1.2, alpha=0.3,
                 linestyle=':', label='Nominal path (S-curve)')

    # Stalled car marker
    ax_traj.add_patch(plt.Rectangle(
        (stalled_x - 2.25, ego_lane_y - 1.0), 4.5, 2.0,
        color='#d62728', alpha=0.85, zorder=5
    ))
    ax_traj.text(stalled_x, ego_lane_y + 1.5, 'Stalled', ha='center', fontsize=8,
                 color='#d62728', fontweight='bold')

    # Moving car initial position marker
    ax_traj.add_patch(plt.Rectangle(
        (moving_x_init - 2.25, middle_lane_y - 1.0), 4.5, 2.0,
        color='#ff7f0e', alpha=0.70, zorder=5
    ))
    ax_traj.text(moving_x_init, middle_lane_y + 1.5, 'Moving\n(t=0)', ha='center', fontsize=7,
                 color='#ff7f0e', fontweight='bold')
    arrow_dx = np.sign(moving_vx) * 8
    ax_traj.annotate('', xy=(moving_x_init + arrow_dx, middle_lane_y),
                     xytext=(moving_x_init, middle_lane_y),
                     arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))

    # Ego trajectories
    for algo in algos:
        d = data[algo]
        x, y = d['x'], d['y']
        ax_traj.plot(x, y, color=colors[algo], linewidth=2.2,
                     linestyle=linestyles[algo], label=labels[algo], zorder=10)

        # Mark backup segments with thicker line
        backup_mask = np.array([m == 'backup' for m in d['mode']])
        if backup_mask.any():
            # Segment contiguous backup regions
            starts = np.where(np.diff(backup_mask.astype(int)) == 1)[0] + 1
            ends   = np.where(np.diff(backup_mask.astype(int)) == -1)[0] + 1
            if backup_mask[0]:
                starts = np.concatenate([[0], starts])
            if backup_mask[-1]:
                ends = np.concatenate([ends, [len(backup_mask)]])
            for s, e in zip(starts, ends):
                ax_traj.plot(x[s:e], y[s:e], color=colors[algo],
                             linewidth=4.0, alpha=0.5, zorder=9)

    ax_traj.set_xlabel('X [m]', fontsize=11)
    ax_traj.set_ylabel('Y [m]', fontsize=11)
    ax_traj.set_title(
        f'(a) Merge Zone Trajectories — stalled x={stalled_x:.0f}m, '
        f'moving car x₀={moving_x_init:+.0f}m, vx={moving_vx:+.1f} m/s',
        fontsize=11, fontweight='bold',
    )
    ax_traj.legend(loc='lower right', fontsize=9, framealpha=0.85)
    x_max = max(max(data[a]['x'].max() for a in algos), stalled_x + 40, moving_x_init + 20)
    ax_traj.set_xlim(-5, min(x_max + 10, track_length))
    ax_traj.set_ylim(backup_lane_y - 3, ego_lane_y + 3)

    # --- Panel (b): h_min over time ---
    for algo in algos:
        d = data[algo]
        ax_h.plot(d['t'], d['h_min'], color=colors[algo], linewidth=1.8,
                  linestyle=linestyles[algo], label=algo)
    ax_h.axhline(0, color='red', linewidth=1.0, alpha=0.6, linestyle='--')
    t_max = max(data[a]['t'].max() for a in algos)
    ax_h.fill_between([0, t_max], 0, -0.5, color='red', alpha=0.12, label='Unsafe region')
    ax_h.set_xlabel('Time [s]', fontsize=10)
    ax_h.set_ylabel(r'$h_{\min}$', fontsize=10)
    ax_h.set_title('(b) Safety Margin $h_{\\min}$', fontsize=10, fontweight='bold')
    ax_h.legend(fontsize=8)

    # --- Panel (c): Mode (nominal / backup) over time ---
    mode_vals = {'nominal': 1, 'backup': 0}
    for i, algo in enumerate(algos):
        d = data[algo]
        mv = np.array([mode_vals[m] for m in d['mode']], dtype=float)
        offset = i * 0.04   # slight vertical offset to distinguish overlapping lines
        ax_mode.plot(d['t'], mv + offset, color=colors[algo], linewidth=2.0,
                     linestyle=linestyles[algo], label=algo, drawstyle='steps-post')
    ax_mode.set_yticks([0, 1])
    ax_mode.set_yticklabels(['Backup', 'Nominal'])
    ax_mode.set_xlabel('Time [s]', fontsize=10)
    ax_mode.set_ylabel('Mode', fontsize=10)
    ax_mode.set_title('(c) Controller Mode Over Time', fontsize=10, fontweight='bold')
    ax_mode.legend(fontsize=8)
    ax_mode.set_ylim(-0.2, 1.3)

    plt.suptitle('Merge Zone Safety Comparison: Gatekeeper vs MPS vs BackupCBF',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {output_path}")
    plt.close(fig)


# =============================================================================
# Main Test Runner
# =============================================================================

def slugify_name(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')
    return slug or "simulation"


def run_test(config: TestConfig) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"  MERGE ZONE: {config.name}")
    print(f"  {config.description}")
    print("=" * 70)

    env, ax, fig = setup_environment(config)
    car, X0, ego_lane_y, backup_lane_y = setup_vehicle(config, env, ax)
    setup_obstacles(config, env)
    mpcc, shielding = setup_controllers(config, car, env, ego_lane_y, backup_lane_y, ax)

    track = config.track
    middle_lane_y = env.get_lane_center(track.middle_lane_idx)
    ref_x, ref_y = build_nominal_path(
        track, config.stalled_x_offset, ego_lane_y, middle_lane_y, track.track_length
    )
    ref_horizon_line, mpc_pred_line = setup_visualization(
        ax, env, ego_lane_y, backup_lane_y, ref_x, ref_y
    )

    simulator = DriftingCarSimulator(car, env, show_animation=True)

    animation_saver = None
    if config.save_animation:
        safe_name = slugify_name(config.name)
        output_dir = f"output/animations/{safe_name}"
        animation_saver = AnimationSaver(output_dir=output_dir, save_per_frame=1, fps=30)
        print(f"\n  Animation saving enabled → {output_dir}/")

    print(f"\nConfiguration:")
    print(f"  Algorithm:      {config.algo_type}")
    print(f"  Stalled car x:  {config.stalled_x_offset:.1f} m (upper lane)")
    print(f"  Moving car x:   {config.moving_x_offset:+.1f} m (middle lane, relative to ego)")
    print(f"  Moving car vx:  {config.moving_vx:+.1f} m/s")
    print(f"  Target speed:   {config.simulation.target_velocity:.1f} m/s")
    print(f"  Expected coll:  {config.expected_collision}")

    results = run_simulation(
        config, car, env, mpcc, shielding, simulator,
        ref_horizon_line, mpc_pred_line, ax, fig, animation_saver,
    )

    if animation_saver is not None:
        animation_saver.export_video(output_name=f"{slugify_name(config.name)}.mp4")

    print("\n" + "-" * 50)
    print("Results:")
    print(f"  Collision:   {'YES' if results['collision'] else 'NO'}")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Nominal:     {results['nominal_steps']} ({100*results['nominal_ratio']:.1f}%)")
    print(f"  Backup:      {results['backup_steps']} ({100*results['backup_ratio']:.1f}%)")
    print(f"  Global min h:{results['global_min_h']:.4f}")

    if results['collision'] == config.expected_collision:
        print(f"\n  ✓ TEST PASSED")
        results['passed'] = True
    else:
        print(f"\n  ✗ TEST FAILED (expected collision={config.expected_collision}, "
              f"got {results['collision']})")
        results['passed'] = False

    print("-" * 50)

    plt.ioff()
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    return results


# =============================================================================
# Test Case Factory
# =============================================================================

def create_merge_zone_test(
    algo_type: str = 'gatekeeper',
    stalled_x_offset: float = 50.0,
    moving_x_offset: float = 80.0,
    moving_vx: float = -10.0,
    save_animation: bool = False,
    expected_collision: bool = False,
) -> TestConfig:
    """
    Create a merge zone test configuration.

    Default scenario: oncoming car in middle lane (moving_vx < 0).
      - Oncoming car starts 80m ahead, travels toward ego at -10 m/s.
      - They meet at t≈4s at x≈41m (right in the S-curve merge zone).
      - Backup (lower lane) initiated at t=0 avoids the car (gap ≈49m when crossing y=0).
      - If backup initiated at t≈4s, the oncoming car is right there → collision.
      → Gatekeeper (full horizon check) commits to backup early and succeeds.
      → MPS (step-by-step) follows nominal until car is adjacent → fails.

    Args:
        algo_type:        'gatekeeper', 'mps', or 'backupcbf'
        stalled_x_offset: x-distance from ego to stalled car in upper lane [m]
        moving_x_offset:  initial x of moving car in middle lane relative to ego [m]
        moving_vx:        x-velocity of moving car [m/s] (negative = oncoming)
        save_animation:   whether to save animation as video
        expected_collision: whether a collision is expected (for pass/fail tracking)
    """
    desc = (
        f"Stalled car at x={stalled_x_offset:.0f}m in upper lane; "
        f"moving car at x={moving_x_offset:+.0f}m, vx={moving_vx:+.1f} m/s "
        f"in middle lane ({algo_type})."
    )
    return TestConfig(
        name=f"MergeZone [algo={algo_type}, stalled={stalled_x_offset:.0f}, "
             f"moving={moving_x_offset:+.0f}, vx={moving_vx:+.0f}]",
        description=desc,
        track=TrackConfig(),
        vehicle=VehicleConfig(mu=1.0),
        simulation=SimulationConfig(),
        stalled_x_offset=stalled_x_offset,
        moving_x_offset=moving_x_offset,
        moving_vx=moving_vx,
        algo_type=algo_type,
        save_animation=save_animation,
        expected_collision=expected_collision,
    )


# =============================================================================
# Sweep Mode
# =============================================================================

def run_sweep(
    stalled_x_offset: float = 50.0,
    moving_offsets: Optional[list] = None,
    moving_vx: float = -10.0,
    algos: Optional[list] = None,
) -> None:
    """
    Sweep moving_x_offset across all algorithms to find the interesting regime.
    Prints a summary table of pass/fail/collision results.
    """
    if moving_offsets is None:
        moving_offsets = [50, 60, 70, 80, 90, 100, 120]
    if algos is None:
        algos = ALGO_TYPES

    print("\n" + "=" * 70)
    print(f"  SWEEP: stalled_x={stalled_x_offset:.0f}m, moving_vx={moving_vx:+.1f} m/s, "
          f"moving_x ∈ {moving_offsets}")
    print("=" * 70)

    # Header
    header = f"{'moving_x':>10}"
    for algo in algos:
        header += f"  {algo:>12}"
    print(header)
    print("-" * (10 + 14 * len(algos)))

    for moving_x in moving_offsets:
        row = f"{moving_x:>+10.0f}"
        for algo in algos:
            config = create_merge_zone_test(
                algo_type=algo,
                stalled_x_offset=stalled_x_offset,
                moving_x_offset=float(moving_x),
                moving_vx=moving_vx,
            )
            res = run_test(config)
            status = "COLLISION" if res['collision'] else "safe"
            row += f"  {status:>12}"
        print(row)

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge zone safety shielding test (Gatekeeper / MPS / BackupCBF)'
    )
    parser.add_argument('--algo', type=str, default='gatekeeper', choices=ALGO_TYPES,
                        help='Shielding algorithm (default: gatekeeper)')
    parser.add_argument('--stalled', type=float, default=50.0,
                        help='x-offset of stalled car in upper lane, m (default: 50)')
    parser.add_argument('--moving', type=float, default=80.0,
                        help='x-offset of moving car in middle lane relative to ego, m '
                             '(default: 80 for oncoming scenario)')
    parser.add_argument('--moving-vx', type=float, default=-10.0,
                        help='vx of moving car in m/s (negative=oncoming, default: -10)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep moving_x_offset across all algorithms')
    parser.add_argument('--sweep-stalled', type=float, default=50.0,
                        help='stalled_x_offset to use during sweep (default: 50)')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    parser.add_argument('--figure', action='store_true',
                        help='Generate trajectory overlay figure for all three algorithms')
    parser.add_argument('--figure-out', type=str, default='output/merge_zone_comparison.png',
                        help='Output path for the trajectory figure (default: output/merge_zone_comparison.png)')
    args = parser.parse_args()

    if args.figure:
        generate_trajectory_figure(
            stalled_x_offset=args.stalled,
            moving_x_offset=args.moving,
            moving_vx=args.moving_vx,
            output_path=args.figure_out,
        )
    elif args.sweep:
        run_sweep(stalled_x_offset=args.sweep_stalled)
    else:
        config = create_merge_zone_test(
            algo_type=args.algo,
            stalled_x_offset=args.stalled,
            moving_x_offset=args.moving,
            moving_vx=args.moving_vx,
            save_animation=args.save,
        )
        run_test(config)


if __name__ == "__main__":
    main()
