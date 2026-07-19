"""
Car-following nominal models and the scripted lead-car velocity profile.

All quantities are 1D longitudinal. A "car" is described by a dict with keys
'x' (position of car center), 'vx' (speed), and optionally 'length' (body
length, default 4.5 m) — the same convention used by `_idm_accel` in
safe_control/envs/drifting_env.py.

Provided functions
------------------
  ovm_accel(follower, leader, params)  -> float
      Optimal Velocity Model (OVM) acceleration.
  nominal_accel(model, follower, leader, params) -> float
      Dispatch to 'ovm' (ovm_accel) or 'idm' (reuses _idm_accel), clipped to a_max.
  lead_velocity(t, params) -> float
      Scripted stop-and-go velocity profile for the lead car.
"""

import numpy as np
from safe_control.envs.drifting_env import _idm_accel


def _bumper_gap(follower: dict, leader: dict) -> float:
    """Bumper-to-bumper gap (matches the convention in _idm_accel)."""
    f_len = float(follower.get('length', follower.get('body_length', 4.5)))
    l_len = float(leader.get('length', leader.get('body_length', 4.5)))
    return (leader['x'] - l_len / 2) - (follower['x'] + f_len / 2)


def ovm_accel(follower: dict, leader: dict, params: dict) -> float:
    """Optimal Velocity Model acceleration.

    v_dot = alpha * (V_h(h) - v) + beta * (W(v_lead) - v)
      h          = bumper-to-bumper gap to the leader
      V_h(h)     = clip(kappa * (h - h_st), 0, v_max)   (constant time headway 1/kappa)
      W(v_lead)  = min(v_lead, v_max)

    Args:
        follower: Dict with 'x', 'vx', optionally 'length'.
        leader:   Dict with 'x', 'vx', optionally 'length'.
        params:   OVM parameters — alpha, beta, kappa, h_st, v_max.

    Returns:
        Longitudinal acceleration [m/s^2].
    """
    v = float(follower.get('vx', 0.0))
    v_l = float(leader.get('vx', 0.0))
    h = _bumper_gap(follower, leader)

    alpha = float(params.get('alpha', 0.6))
    beta = float(params.get('beta', 0.9))
    kappa = float(params.get('kappa', 0.6))
    h_st = float(params.get('h_st', 5.0))
    v_max = float(params.get('v_max', 12.0))

    V_h = float(np.clip(kappa * (h - h_st), 0.0, v_max))
    W = min(v_l, v_max)
    return alpha * (V_h - v) + beta * (W - v)


def nominal_accel(model: str, follower: dict, leader: dict, params: dict) -> float:
    """Nominal car-following acceleration, dispatched by model name.

    Used as the ego's nominal controller (follower=ego, leader=lead car) and is
    a generic follower-responds-to-leader model.

    Args:
        model: 'ovm' (OVM) or 'idm' (reuses safe_control _idm_accel).
        follower, leader: car dicts as above.
        params: model parameters; 'a_max' (max acceleration a_bar) and, if present,
                'a_e' (max deceleration) set the actuator limits.

    Returns:
        Acceleration clipped to the actuator limits [-a_e, a_max].  The braking
        bound matches the vehicle's real capability a_e (default = a_max), so an
        unfiltered collision reflects the model's sluggish *response*, not an
        artificially low brake cap.
    """
    if model == 'ovm':
        a = ovm_accel(follower, leader, params)
    elif model == 'idm':
        a = _idm_accel(follower, leader, params)
    else:
        raise ValueError(f"Unknown nominal model '{model}' (expected 'ovm' or 'idm')")

    a_accel = float(params.get('a_max', 3.0))
    a_brake = float(params.get('a_e', a_accel))
    return float(np.clip(a, -a_brake, a_accel))


def lead_velocity(t: float, params: dict) -> float:
    """Scripted stop-and-go velocity profile for the lead car.

    Phases (piecewise-linear in time, clipped at >= 0):
      [0, t_cruise)            cruise at v_cruise
      [t_cruise, t_brake_end)  hard brake (decel a_brake) down to a stop
      [t_brake_end, t_hold)    hold at standstill
      [t_hold, t_accel_end)    re-accelerate (accel a_accel) back to v_cruise
      [t_accel_end, inf)       cruise at v_cruise

    Args:
        t: time [s].
        params: profile parameters — v_cruise, t_cruise, a_brake, t_hold_dur,
                a_accel.

    Returns:
        Lead-car speed [m/s] (>= 0).
    """
    v_cruise = float(params.get('v_cruise', 10.0))
    t_cruise = float(params.get('t_cruise', 3.0))
    a_brake = float(params.get('a_brake', 5.0))     # magnitude of deceleration
    t_hold_dur = float(params.get('t_hold_dur', 3.0))
    a_accel = float(params.get('a_accel', 2.0))

    t_brake_dur = v_cruise / a_brake
    t_brake_end = t_cruise + t_brake_dur
    t_hold_end = t_brake_end + t_hold_dur
    t_accel_dur = v_cruise / a_accel
    t_accel_end = t_hold_end + t_accel_dur

    if t < t_cruise:
        v = v_cruise
    elif t < t_brake_end:
        v = v_cruise - a_brake * (t - t_cruise)
    elif t < t_hold_end:
        v = 0.0
    elif t < t_accel_end:
        v = a_accel * (t - t_hold_end)
    else:
        v = v_cruise
    return float(max(v, 0.0))
