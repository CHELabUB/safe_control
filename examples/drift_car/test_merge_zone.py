"""
Created on June 20th, 2026
@author: Chaozhe He

@description:
Merge zone test case for safety shielding algorithms (Gatekeeper, MPS, BackupCBF).

Three-vehicle scenario on a 3-lane straight track:
  - Ego vehicle:   upper lane (lane 0, y=4), initial x=1, travels at target_velocity
  - Stalled car:   upper lane (lane 0), x = stalled_x_offset ahead of ego (static)
  - Moving car:    middle lane (lane 1, y=0), x = moving_x_offset relative to ego,
                   travels at moving_vx (default: faster than ego, catches up)
  - Lower lane (lane 2, y=-4): empty; not used as backup in this version

Nominal plan (MPCC): S-curve reference path upper→middle (around stalled car)→upper
Backup policy: ABORT — LaneChangeController returning to upper lane (y=4)

Abort backup safety constraint:
  Safe abort requires: x_ego + v·T_b < stalled_x - r_combined
  With T_b=1.5s, v=10m/s, stalled_x=80m: x_ego < 62.8m
  Merge starts at merge_start = max(5, stalled_x-30) = 50m
  → abort window extends 12.8m into the merge zone (safe partial abort)

Gets-stuck analysis:
  After abort completes (ego back at y=4), algorithms re-evaluate every step.
  Once moving car clears the middle lane, nominal S-curve is certifiable again.
  Deadlock only if abort triggered at x > 62.8m (abort collides with stalled car).

Usage:
    uv run python examples/drift_car/test_merge_zone.py --algo gatekeeper
    uv run python examples/drift_car/test_merge_zone.py --algo mps
    uv run python examples/drift_car/test_merge_zone.py --algo backupcbf
    uv run python examples/drift_car/test_merge_zone.py --figure
    uv run python examples/drift_car/test_merge_zone.py --sweep
    uv run python examples/drift_car/test_merge_zone.py --algo gatekeeper --save

@required-scripts: safe_control/shielding/gatekeeper.py, safe_control/shielding/mps.py
"""

import csv
import os
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union, List
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


def _code_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return 'unknown'


def _build_run_id(algo_type: str, stalled_x: float, moving_x: float, moving_vx: float) -> str:
    """Stable, human-readable identifier for one (algo, scenario) combination."""
    # Encode sign explicitly so +10 and -10 don't collide after slugification
    mx_sign = 'n' if moving_x < 0 else 'p'
    vx_sign = 'n' if moving_vx < 0 else 'p'
    return (
        f"mergezone_{algo_type}"
        f"_stalled{stalled_x:.0f}"
        f"_moving{mx_sign}{abs(moving_x):.0f}"
        f"_vx{vx_sign}{abs(moving_vx):.0f}"
    )


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
    backup_horizon_time: float = 1.5
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
    stalled_x_offset: float = 80.0   # x of stalled car in upper lane (relative to ego)
    moving_x_offset: float = -10.0   # x of moving car in middle lane (relative to ego)
    moving_vx: float = 12.0          # vx of moving car (negative = oncoming, positive = same dir)
    algo_type: str = 'gatekeeper'
    save_animation: bool = False
    expected_collision: bool = False


# =============================================================================
# Run Registry
# =============================================================================

@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    algo: str
    stalled_x: float
    moving_x: float
    moving_vx: float
    backup_horizon: float
    nominal_horizon: float
    collision: str        # 'YES' / 'NO'
    total_steps: int
    nominal_pct: float
    global_min_h: float
    animation_path: str   # relative path to .mp4, or ''
    data_path: str        # relative path to .npz, or ''
    code_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id':          self.run_id,
            'timestamp':       self.timestamp,
            'algo':            self.algo,
            'stalled_x':       f"{self.stalled_x:.1f}",
            'moving_x':        f"{self.moving_x:.1f}",
            'moving_vx':       f"{self.moving_vx:.1f}",
            'backup_horizon':  f"{self.backup_horizon:.2f}",
            'nominal_horizon': f"{self.nominal_horizon:.2f}",
            'collision':       self.collision,
            'total_steps':     str(self.total_steps),
            'nominal_pct':     f"{self.nominal_pct:.1f}",
            'global_min_h':    f"{self.global_min_h:.4f}",
            'animation_path':  self.animation_path,
            'data_path':       self.data_path,
            'code_hash':       self.code_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunRecord':
        return cls(
            run_id=d['run_id'],
            timestamp=d['timestamp'],
            algo=d['algo'],
            stalled_x=float(d['stalled_x']),
            moving_x=float(d['moving_x']),
            moving_vx=float(d['moving_vx']),
            backup_horizon=float(d['backup_horizon']),
            nominal_horizon=float(d['nominal_horizon']),
            collision=d['collision'],
            total_steps=int(d['total_steps']),
            nominal_pct=float(d['nominal_pct']),
            global_min_h=float(d['global_min_h']),
            animation_path=d.get('animation_path', ''),
            data_path=d.get('data_path', ''),
            code_hash=d.get('code_hash', ''),
        )


class RunRegistry:
    CSV_PATH = "output/run_registry.csv"
    FIELDS = [
        'run_id', 'timestamp', 'algo', 'stalled_x', 'moving_x', 'moving_vx',
        'backup_horizon', 'nominal_horizon', 'collision', 'total_steps',
        'nominal_pct', 'global_min_h', 'animation_path', 'data_path', 'code_hash',
    ]

    def load(self) -> List[RunRecord]:
        if not os.path.exists(self.CSV_PATH):
            return []
        with open(self.CSV_PATH, newline='') as f:
            return [RunRecord.from_dict(row) for row in csv.DictReader(f)]

    def upsert(self, record: RunRecord) -> None:
        os.makedirs(os.path.dirname(self.CSV_PATH), exist_ok=True)
        rows: Dict[str, RunRecord] = {r.run_id: r for r in self.load()}
        rows[record.run_id] = record
        with open(self.CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            for r in rows.values():
                writer.writerow(r.to_dict())

    def find_stale(self) -> List[RunRecord]:
        current_hash = _code_hash()
        stale = []
        for r in self.load():
            missing_animation = r.animation_path and not os.path.exists(r.animation_path)
            outdated_code = r.code_hash not in ('unknown', current_hash)
            if missing_animation or outdated_code:
                stale.append(r)
        return stale

    def print_table(self) -> None:
        rows = self.load()
        if not rows:
            print("  (registry is empty — run a simulation first)")
            return
        cols = ['run_id', 'timestamp', 'collision', 'total_steps',
                'nominal_pct', 'global_min_h', 'animation_path', 'code_hash']
        dicts = [r.to_dict() for r in rows]
        widths = {c: max(len(c), max(len(str(d.get(c, ''))) for d in dicts)) for c in cols}
        sep = '  '
        header = sep.join(f"{c:{widths[c]}}" for c in cols)
        divider = sep.join('-' * widths[c] for c in cols)
        print(header)
        print(divider)
        for d in dicts:
            print(sep.join(f"{str(d.get(c, '')):{widths[c]}}" for c in cols))


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

    # Backup: abort — steer back to original upper lane (y=ego_lane_y)
    backup_controller = LaneChangeController(car.robot_spec, sim.dt, direction='left')
    print(f"  Backup: abort to upper lane (y={ego_lane_y:.2f})")

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

    shielding.set_backup_controller(backup_controller, target=ego_lane_y)
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
    # Abort target = upper lane (same as ego start lane)
    ax.axhline(y=ego_lane_y, color='cyan', linewidth=1.5, alpha=0.5,
               linestyle=':', label='Upper lane / abort target')

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
    stalled_x_offset: float = 80.0,
    moving_x_offset: float = -10.0,
    moving_vx: float = 12.0,
) -> Dict[str, Any]:
    """
    Run simulation headlessly and return full per-step trajectory data.

    Results are cached in output/data/{run_id}.npz keyed by git hash.
    On a second call with the same parameters and unchanged code the cache is
    loaded directly — no re-simulation needed.
    """
    run_id = _build_run_id(algo_type, stalled_x_offset, moving_x_offset, moving_vx)
    data_path = f"output/data/{run_id}.npz"
    current_hash = _code_hash()

    # --- Cache hit? ---
    if os.path.exists(data_path):
        try:
            npz = np.load(data_path, allow_pickle=True)
            if str(npz['code_hash']) == current_hash:
                print(f"  Loading cached trajectory: {algo_type} ({data_path})")
                return {
                    'algo':         str(npz['algo']),
                    'x':            npz['x'],
                    'y':            npz['y'],
                    'theta':        npz['theta'] if 'theta' in npz else np.zeros(len(npz['x'])),
                    'mode':         list(npz['mode']),
                    'h_min':        npz['h_min'],
                    't':            npz['t'],
                    'collision':    bool(npz['collision']),
                    'ego_lane_y':   float(npz['ego_lane_y']),
                    'backup_lane_y': float(npz['backup_lane_y']),
                    'middle_lane_y': float(npz['middle_lane_y']),
                    'stalled_x':    float(npz['stalled_x']),
                    'moving_x':     float(npz['moving_x']),
                    'moving_vx':    float(npz['moving_vx']),
                    'track_length': float(npz['track_length']),
                    'run_id':       run_id,
                    'data_path':    data_path,
                }
        except Exception:
            pass  # corrupted cache — fall through to re-simulate

    # --- Simulate ---
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
    middle_lane_y = env.get_lane_center(track.middle_lane_idx)

    xs, ys, thetas, modes, h_mins, times = [], [], [], [], [], []
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
        thetas.append(float(state[2]))   # heading angle
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

    # --- Save cache ---
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(
        data_path,
        algo=algo_type,
        x=np.array(xs),
        y=np.array(ys),
        theta=np.array(thetas),
        mode=np.array(modes),
        h_min=np.array(h_mins),
        t=np.array(times),
        collision=collision,
        ego_lane_y=ego_lane_y,
        backup_lane_y=backup_lane_y,
        middle_lane_y=middle_lane_y,
        stalled_x=stalled_x_offset,
        moving_x=moving_x_offset,
        moving_vx=moving_vx,
        track_length=float(track.track_length),
        code_hash=current_hash,
    )

    return {
        'algo':         algo_type,
        'x':            np.array(xs),
        'y':            np.array(ys),
        'theta':        np.array(thetas),
        'mode':         modes,
        'h_min':        np.array(h_mins),
        't':            np.array(times),
        'collision':    collision,
        'ego_lane_y':   ego_lane_y,
        'backup_lane_y': backup_lane_y,
        'middle_lane_y': middle_lane_y,
        'stalled_x':    stalled_x_offset,
        'moving_x':     moving_x_offset,
        'moving_vx':    moving_vx,
        'track_length': track.track_length,
        'run_id':       run_id,
        'data_path':    data_path,
    }


def _draw_car_at(ax, cx, cy, theta, body_color,
                 tire_color=(0.25, 0.25, 0.25), alpha=0.90, zorder=8,
                 a=1.4, b=1.4, body_length=4.5, body_width=2.0):
    """
    Draw a realistic car silhouette (body polygon + 4 tire rectangles) at (cx, cy).

    Uses the same 10-point polygon geometry as DriftingEnv._create_obstacle_car_patches().
    """
    L, W = body_length, body_width
    rear_overhang  = (L - a - b) * 0.4
    front_overhang = (L - a - b) * 0.6

    # 10-point tapered car body (local frame, x forward)
    body_verts = np.array([
        [-b - rear_overhang,        -W / 2       ],
        [-b - rear_overhang,         W / 2       ],
        [-b - rear_overhang + 0.3,   W / 2 + 0.05],
        [ a + front_overhang - 0.8,  W / 2 + 0.05],
        [ a + front_overhang - 0.3,  W / 2 * 0.7 ],
        [ a + front_overhang,        W / 2 * 0.5 ],
        [ a + front_overhang,       -W / 2 * 0.5 ],
        [ a + front_overhang - 0.3, -W / 2 * 0.7 ],
        [ a + front_overhang - 0.8, -W / 2 - 0.05],
        [-b - rear_overhang + 0.3,  -W / 2 - 0.05],
    ]).T   # shape (2, 10)

    # Tire rectangles (local frame)
    tl, tw    = 0.6 / 2, 0.25 / 2
    ty_off    = W / 2 - tw * 2 - 0.1
    tire_ctrs = {
        'fl': np.array([ a,  ty_off]),
        'fr': np.array([ a, -ty_off]),
        'rl': np.array([-b,  ty_off]),
        'rr': np.array([-b, -ty_off]),
    }
    tire_verts = np.array([[-tl, -tw], [-tl, tw], [tl, tw], [tl, -tw]]).T  # (2, 4)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    center = np.array([[cx], [cy]])

    # Body
    body_world = R @ body_verts + center
    ax.add_patch(MplPolygon(
        body_world.T, closed=True,
        facecolor=body_color, edgecolor='black',
        linewidth=0.8, alpha=alpha, zorder=zorder,
    ))

    # Tires
    for pos in tire_ctrs.values():
        pos_world = R @ pos.reshape(2, 1) + center
        tire_world = R @ tire_verts + pos_world
        ax.add_patch(MplPolygon(
            tire_world.T, closed=True,
            facecolor=tire_color, edgecolor='black',
            linewidth=0.5, alpha=alpha, zorder=zorder + 1,
        ))


def generate_trajectory_figure(
    stalled_x_offset: float = 80.0,
    moving_x_offset: float = -10.0,
    moving_vx: float = 12.0,
    output_path: str = 'output/merge_zone_abort_comparison.png',
) -> None:
    """
    Trajectory overlay figure whose panel (a) uses DriftingEnv.setup_plot() and
    DriftingEnv._create_obstacle_car_patches() — the exact same rendering as the video.

    Panel (a): Road/grass/lanes from DriftingEnv; car-body snapshots every 1 s.
    Panel (b): h_backup safety margin over time per algorithm.
    Panel (c): Backup activation intervals as filled bands per algorithm.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    algos = ['backupcbf', 'mps', 'gatekeeper']
    colors = {
        'backupcbf':  np.array([0.12, 0.47, 0.71]),
        'mps':        np.array([0.58, 0.40, 0.74]),
        'gatekeeper': np.array([0.17, 0.63, 0.17]),
    }
    labels = {
        'backupcbf':  'Ego Car: Backup CBF',
        'mps':        'Ego Car: MPS',
        'gatekeeper': 'Ego Car: Gatekeeper',
    }
    linestyles = {'backupcbf': '-', 'mps': '--', 'gatekeeper': '-.'}
    CAR_BODY_INTERVAL = 20   # one snapshot per 1 s (dt=0.05, 20 steps)
    obstacle_spec = {'body_length': 4.5, 'body_width': 2.0, 'a': 1.4, 'b': 1.4, 'radius': 1.0}

    print(f"\nGenerating trajectory comparison figure ...")
    print(f"  Scenario: stalled_x={stalled_x_offset:.0f}m, "
          f"moving_x={moving_x_offset:+.0f}m, moving_vx={moving_vx:+.1f} m/s")

    data = {}
    for algo in algos:
        data[algo] = collect_trajectory_data(algo, stalled_x_offset, moving_x_offset, moving_vx)

    ref_data      = data[algos[0]]
    ego_lane_y    = ref_data['ego_lane_y']
    backup_lane_y = ref_data['backup_lane_y']
    middle_lane_y = ref_data['middle_lane_y']
    stalled_x     = ref_data['stalled_x']
    moving_x_init = ref_data['moving_x']
    track_length  = ref_data['track_length']
    half_lw       = 2.0

    tc = TrackConfig()
    ref_px, ref_py = build_nominal_path(tc, stalled_x, ego_lane_y, middle_lane_y, track_length)

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    plt.ioff()
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[3, 1, 1],
        hspace=0.50, top=0.91, bottom=0.07, left=0.06, right=0.97,
    )
    ax_traj = fig.add_subplot(gs[0])
    ax_h    = fig.add_subplot(gs[1])
    ax_mode = fig.add_subplot(gs[2])

    # -------------------------------------------------------------------------
    # Panel (a): use DriftingEnv's actual rendering — road, grass, lane dividers
    # -------------------------------------------------------------------------
    road_env = DriftingEnv(
        track_type='straight',
        track_width=tc.lane_width * tc.num_lanes,
        track_length=track_length,
        num_lanes=tc.num_lanes,
    )
    road_env.setup_plot(ax=ax_traj, fig=fig)   # draws grass + road surface + lane markings

    # Override equal-aspect lock set by setup_plot, and set the desired view
    ax_traj.set_aspect('auto')
    ax_traj.grid(False)
    x_lo = min(moving_x_init - 8, -8)
    x_hi = min(max(data[a]['x'].max() for a in algos) + 15, track_length)
    ax_traj.set_xlim(x_lo, x_hi)
    ax_traj.set_ylim(backup_lane_y - 1.2, ego_lane_y + 2.0)

    # Faint nominal S-curve reference
    ax_traj.plot(ref_px, ref_py, color='yellow', linewidth=1.0,
                 linestyle=':', alpha=0.35, zorder=3)

    # Stalled car — same call as DriftingEnv uses internally for static obstacles
    for p in road_env._create_obstacle_car_patches(
            {'x': stalled_x, 'y': ego_lane_y, 'theta': 0.0, 'spec': obstacle_spec},
            body_color=(0.72, 0.15, 0.15), zorder=12):
        ax_traj.add_patch(p)
    ax_traj.text(stalled_x, ego_lane_y + half_lw + 0.25, 'Obstacle',
                 ha='center', va='bottom', fontsize=8,
                 color='#e84040', fontweight='bold', zorder=13)

    # Moving car at t=0 — same call used for dynamic obstacles
    for p in road_env._create_obstacle_car_patches(
            {'x': moving_x_init, 'y': middle_lane_y, 'theta': 0.0, 'spec': obstacle_spec},
            body_color=(0.85, 0.50, 0.10), zorder=11):
        ax_traj.add_patch(p)
    ax_traj.annotate(
        '', xy=(moving_x_init + np.sign(moving_vx) * 6, middle_lane_y),
        xytext=(moving_x_init, middle_lane_y),
        arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2.0), zorder=14,
    )

    # Ego car snapshots — same geometry via _create_obstacle_car_patches
    legend_patches = []
    for algo in algos:
        d = data[algo]
        x, y, theta_arr = d['x'], d['y'], d['theta']

        # Trajectory trace (same as DriftingCar.trajectory_line)
        ax_traj.plot(x, y, color=colors[algo], linewidth=1.5,
                     linestyle=linestyles[algo], alpha=0.55, zorder=4)

        for i in range(0, len(x), CAR_BODY_INTERVAL):
            is_backup = (d['mode'][i] == 'backup')
            for p in road_env._create_obstacle_car_patches(
                    {'x': x[i], 'y': y[i], 'theta': float(theta_arr[i]), 'spec': obstacle_spec},
                    body_color=tuple(colors[algo]), zorder=10):
                p.set_alpha(0.90 if not is_backup else 0.50)
                ax_traj.add_patch(p)

        legend_patches.append(mpatches.Patch(
            facecolor=colors[algo], edgecolor='black', linewidth=0.8, label=labels[algo],
        ))

    legend_patches += [
        mpatches.Patch(facecolor=(0.72, 0.15, 0.15), edgecolor='black',
                       linewidth=0.8, label='Obstacle'),
        mpatches.Patch(facecolor=(0.85, 0.50, 0.10), edgecolor='black',
                       linewidth=0.8, label='Moving car (t=0)'),
    ]
    ax_traj.set_xlabel('X [m]', fontsize=11)
    ax_traj.set_ylabel('Y [m]', fontsize=11)
    ax_traj.set_title(
        f'(a) Merge Zone Trajectories — stalled x={stalled_x:.0f} m, '
        f'moving car x₀={moving_x_init:+.0f} m, vx={moving_vx:+.1f} m/s',
        fontsize=11, fontweight='bold',
    )
    ax_traj.legend(handles=legend_patches, loc='upper left',
                   fontsize=8, framealpha=0.75, ncol=3)

    # -------------------------------------------------------------------------
    # Panel (b): h_backup over time
    # -------------------------------------------------------------------------
    t_max = max(data[a]['t'].max() for a in algos)
    h_all = np.concatenate([data[a]['h_min'] for a in algos])
    h_lo  = min(h_all.min() - 0.3, -0.5)
    h_hi  = max(h_all.max() + 0.2,  2.0)

    ax_h.fill_between([0, t_max], h_lo, 0, color='#d62728', alpha=0.18, zorder=1)
    ax_h.axhline(0, color='#d62728', linewidth=1.0, linestyle='--', alpha=0.7, zorder=2)
    ax_h.text(t_max * 0.98, h_lo * 0.55, 'Unsafe Region',
              ha='right', va='center', fontsize=8, color='#d62728', alpha=0.8)
    for algo in algos:
        d = data[algo]
        ax_h.plot(d['t'], d['h_min'], color=colors[algo], linewidth=1.8,
                  linestyle=linestyles[algo], label=algo, zorder=3)
    ax_h.set_xlim(0, t_max)
    ax_h.set_ylim(h_lo, h_hi)
    ax_h.set_xlabel('Time [s]', fontsize=10)
    ax_h.set_ylabel(r'$h_{\mathrm{backup}}$', fontsize=10)
    ax_h.set_title('(b) Backup CBF Value $h_{\\mathrm{backup}}$', fontsize=10, fontweight='bold')
    ax_h.legend(fontsize=8, loc='upper right')
    ax_h.grid(True, alpha=0.25, linestyle=':')

    # -------------------------------------------------------------------------
    # Panel (c): Backup activation intervals
    # -------------------------------------------------------------------------
    row_height, row_gap = 0.7, 0.15
    yticks, ytick_labels = [], []
    for i, algo in enumerate(algos):
        d = data[algo]
        row_y   = i * (row_height + row_gap)
        t_arr   = d['t']
        bk_mask = np.array([m == 'backup' for m in d['mode']])
        ax_mode.fill_between([t_arr[0], t_arr[-1]], row_y, row_y + row_height,
                             color=colors[algo], alpha=0.15)
        starts = np.where(np.diff(bk_mask.astype(int)) == 1)[0] + 1
        ends   = np.where(np.diff(bk_mask.astype(int)) == -1)[0] + 1
        if bk_mask[0]:
            starts = np.concatenate([[0], starts])
        if bk_mask[-1]:
            ends = np.concatenate([ends, [len(bk_mask)]])
        for s, e in zip(starts, ends):
            ax_mode.fill_between(
                [t_arr[s], t_arr[min(e, len(t_arr) - 1)]],
                row_y, row_y + row_height, color=colors[algo], alpha=0.85,
            )
        yticks.append(row_y + row_height / 2)
        ytick_labels.append(algo)
    ax_mode.set_xlim(0, t_max)
    ax_mode.set_ylim(-0.1, len(algos) * (row_height + row_gap))
    ax_mode.set_yticks(yticks)
    ax_mode.set_yticklabels(ytick_labels, fontsize=9)
    ax_mode.set_xlabel('Time [s]', fontsize=10)
    ax_mode.set_title('(c) Backup Activation Intervals  (solid = backup, faint = nominal)',
                      fontsize=10, fontweight='bold')
    ax_mode.grid(axis='x', alpha=0.25, linestyle=':')

    plt.suptitle(
        'Merge Zone Safety Comparison: Gatekeeper vs MPS vs Backup CBF',
        fontsize=13, fontweight='bold', y=0.97,
    )
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

    run_id = _build_run_id(
        config.algo_type, config.stalled_x_offset,
        config.moving_x_offset, config.moving_vx,
    )

    animation_saver = None
    animation_path = ''
    if config.save_animation:
        output_dir = f"output/animations/{run_id}"
        animation_saver = AnimationSaver(output_dir=output_dir, save_per_frame=1, fps=30)
        animation_path = f"{output_dir}/{run_id}.mp4"
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
        animation_saver.export_video(output_name=f"{run_id}.mp4")

    # --- Persist run record ---
    sim = config.simulation
    registry = RunRegistry()
    record = RunRecord(
        run_id=run_id,
        timestamp=datetime.now().isoformat(timespec='seconds'),
        algo=config.algo_type,
        stalled_x=config.stalled_x_offset,
        moving_x=config.moving_x_offset,
        moving_vx=config.moving_vx,
        backup_horizon=sim.backup_horizon_time,
        nominal_horizon=sim.nominal_horizon_time,
        collision='YES' if results['collision'] else 'NO',
        total_steps=results['total_steps'],
        nominal_pct=100.0 * results['nominal_ratio'],
        global_min_h=results['global_min_h'],
        animation_path=animation_path,
        data_path='',
        code_hash=_code_hash(),
    )
    registry.upsert(record)

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
    stalled_x_offset: float = 80.0,
    moving_x_offset: float = -10.0,
    moving_vx: float = 12.0,
    save_animation: bool = False,
    expected_collision: bool = False,
) -> TestConfig:
    """
    Create a merge zone test configuration.

    Default scenario: catching-up car in middle lane (abort-to-upper-lane backup).
      - Stalled car at x=80m in upper lane (ego's home lane).
      - Moving car starts 10m behind ego in middle lane, travels at 12 m/s (ego at 10 m/s).
        Car catches up to ego at t≈5.5s when ego is executing the S-curve merge (x≈56m).
      - Backup policy: abort the lane change, return to upper lane (y=+4).
        Safe abort window: ego must be at x < 62.8m when abort triggers (T_b=1.5s, S=80).
        This gives 12.8m of merge overlap — the abort remains safe while partially merged.

    Backup geometry constraint:
      safe_abort if:  x_ego + v * T_b < stalled_x - r_combined
                      x_ego + 15 < 77.8  →  x_ego < 62.8 m
      merge_start = max(5, stalled_x-30) = 50 m
      → abort is safe for up to 12.8 m into the merge zone.

    "Gets stuck?" analysis:
      After a successful abort (ego returns to y=4), algorithms re-evaluate at every step.
      Once the moving car clears the middle lane, the nominal S-curve is certifiable again
      and ego resumes it automatically. True deadlock only if abort triggered AFTER x=62.8m
      (abort itself collides with stalled car) — prevented by the T_b=1.5s horizon.

    Args:
        algo_type:        'gatekeeper', 'mps', or 'backupcbf'
        stalled_x_offset: x-distance from ego to stalled car in upper lane [m]
        moving_x_offset:  initial x of moving car in middle lane relative to ego [m]
        moving_vx:        x-velocity of moving car [m/s] (positive = same direction)
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
    parser.add_argument('--stalled', type=float, default=80.0,
                        help='x-offset of stalled car in upper lane, m (default: 80)')
    parser.add_argument('--moving', type=float, default=-10.0,
                        help='x-offset of moving car in middle lane relative to ego, m '
                             '(default: -10 for catching-up scenario)')
    parser.add_argument('--moving-vx', type=float, default=12.0,
                        help='vx of moving car in m/s (positive=same dir, default: 12)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep moving_x_offset across all algorithms')
    parser.add_argument('--sweep-stalled', type=float, default=80.0,
                        help='stalled_x_offset to use during sweep (default: 80)')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    parser.add_argument('--figure', action='store_true',
                        help='Generate trajectory overlay figure for all three algorithms')
    parser.add_argument('--figure-out', type=str, default='output/merge_zone_comparison.png',
                        help='Output path for the trajectory figure (default: output/merge_zone_comparison.png)')
    parser.add_argument('--list', action='store_true',
                        help='Print the run registry table and exit')
    parser.add_argument('--regen-stale', action='store_true',
                        help='Re-run any registry entries whose output files are missing or code has changed')
    args = parser.parse_args()

    if args.list:
        registry = RunRegistry()
        print(f"\nRun registry ({RunRegistry.CSV_PATH}):")
        registry.print_table()
    elif args.regen_stale:
        registry = RunRegistry()
        stale = registry.find_stale()
        if not stale:
            print("All registry entries are up-to-date.")
        else:
            print(f"Found {len(stale)} stale entry/entries — regenerating...")
            for record in stale:
                print(f"\n  → {record.run_id}")
                config = create_merge_zone_test(
                    algo_type=record.algo,
                    stalled_x_offset=record.stalled_x,
                    moving_x_offset=record.moving_x,
                    moving_vx=record.moving_vx,
                    save_animation=bool(record.animation_path),
                )
                run_test(config)
    elif args.figure:
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
