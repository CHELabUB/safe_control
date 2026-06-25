"""
Helper models for the rear-aware (three-car / ego+rear) examples.

Reuses the car-following models from examples/car_following:
  - nominal_accel(model, follower, leader, params)  (OVM / IDM dispatch)
  - ovm_accel, _idm_accel

Two roles appear here:
  - the EGO's nominal that brings it to a stop at a target, modeled as OVM against a
    virtual stationary "wall" leader at the stop location;
  - the REAR vehicle, a plain (unfiltered) car-following follower whose leader is the
    ego.

All vehicles are 1D (DoubleIntegrator1D); a "car" is a dict {'x','vx','length'}.
"""

import os
import sys

# car_following modules (shared) on the path.
_CF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'car_following')
sys.path.insert(0, os.path.abspath(_CF_DIR))

from car_following_models import nominal_accel, ovm_accel  # noqa: E402,F401


def ego_stop_nominal(ego: dict, stop_wall_x: float, params: dict,
                     model: str = 'ovm') -> float:
    """Ego nominal acceleration: car-following against a virtual stopped wall.

    The car-following model (`model`) decelerates the ego to rest a short distance
    before the wall, so `stop_wall_x` acts as the desired stopping location.

    Args:
        ego: ego car dict {'x','vx','length'}.
        stop_wall_x: position of the virtual stationary leader [m].
        params: model params (OVM and IDM keys; see nominal_accel).
        model: 'ovm' or 'idm'.
    """
    wall = {'x': float(stop_wall_x), 'vx': 0.0, 'length': 0.0}
    return nominal_accel(model, ego, wall, params)


def rear_accel(rear: dict, ego: dict, model: str, params: dict) -> float:
    """Rear vehicle car-following acceleration; its leader is the ego (no filter).

    Args:
        rear: rear car dict {'x','vx','length'}.
        ego:  ego car dict (the rear's leader).
        model: 'ovm' or 'idm'.
        params: the rear's car-following params (may differ from what the ego
                *assumes* about the rear inside its backup CBF).
    """
    return nominal_accel(model, rear, ego, params)
