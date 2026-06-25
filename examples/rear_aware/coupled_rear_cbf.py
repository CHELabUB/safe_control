"""
Coupled (interaction-consistent) HOCBF for rear-end avoidance.

Unlike the backup CBF (which rolls out a counterfactual escape and assumes the rear
reacts to *that*), this controller models the rear vehicle as part of the
closed-loop dynamics and constrains the **applied** ego acceleration so the actual
interaction stays safe. The rear's reaction therefore matches what it really sees.

Coupled state  x = [h_r, v_ego, v_rear]:
    h_r_dot   = v_ego - v_rear
    v_ego_dot = a                         (control,  a in [-a_e, a_acc])
    v_rear_dot = a_r(x)                    (the rear's car-following reaction)

Barrier b = h_r - d_min has relative degree 2 (a enters via v_ego_dot), so a
high-order CBF is used:
    psi1 = (v_ego - v_rear) + alpha1 * (h_r - d_min)
    HOCBF:  a >= a_r(x) - (alpha1 + alpha2)(v_ego - v_rear) - alpha1*alpha2*(h_r - d_min)

This is a single lower bound on the ego acceleration -> the filter is analytic
(no rollout, no QP). a_r(x) is evaluated from the actual joint state.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'car_following')))

from car_following_models import nominal_accel          # noqa: E402

BODY_LENGTH = 4.5


class CoupledRearCBF:
    """Interaction-consistent rear-end HOCBF filter for the ego (1D)."""

    def __init__(self, rear_model, rear_params, alpha1=1.0, alpha2=1.0,
                 d_min=1.0, a_e=3.0, a_acc=3.0, robust_factor=1.0):
        """
        Args:
            rear_model, rear_params: the ego's model of the rear ('ovm'/'idm');
                used to evaluate the rear's reaction a_r(x).
            alpha1, alpha2: HOCBF class-K gains.
            d_min: rear collision margin.
            a_e, a_acc: ego braking / acceleration limits (a in [-a_e, a_acc]).
            robust_factor: in (0, 1]. < 1 robustifies against an under-responsive
                rear: the worst-case reaction is evaluated with the rear's response
                gains (alpha, beta) scaled down by this factor (the rear may brake
                *less* than assumed), giving a larger a_r and a more conservative
                bound. 1.0 = deterministic (trust the assumed model).
        """
        self.rear_model = rear_model
        self.rear_params = dict(rear_params)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.d_min = float(d_min)
        self.a_e = float(a_e)
        self.a_acc = float(a_acc)
        self.robust_factor = float(robust_factor)
        # Worst-case (least-responsive) rear params: smaller alpha/beta -> brakes less.
        self.rear_params_worst = dict(rear_params)
        for k in ('alpha', 'beta'):
            if k in self.rear_params_worst:
                self.rear_params_worst[k] *= self.robust_factor
        self.last_h = None
        self.last_intervened = False

    def rear_accel(self, h, v_ego, v_rear, worst=False):
        """Rear's car-following acceleration at gap h, with the ego as its leader."""
        # Build dicts with a bumper gap equal to h (rear at 0, ego at h + L).
        rear_d = {'x': 0.0, 'vx': float(v_rear), 'length': BODY_LENGTH}
        ego_d = {'x': float(h) + BODY_LENGTH, 'vx': float(v_ego), 'length': BODY_LENGTH}
        params = self.rear_params_worst if worst else self.rear_params
        return nominal_accel(self.rear_model, rear_d, ego_d, params)

    def lower_bound(self, h, v_ego, v_rear, a_r=None):
        """HOCBF lower bound on the ego acceleration (worst-case rear if robust)."""
        if a_r is None:
            a_r = self.rear_accel(h, v_ego, v_rear,
                                  worst=(self.robust_factor < 1.0))
        rel_v = v_ego - v_rear
        return (a_r - (self.alpha1 + self.alpha2) * rel_v
                - self.alpha1 * self.alpha2 * (h - self.d_min))

    def filter(self, u_nom, h, v_ego, v_rear):
        """Return the safe ego acceleration closest to u_nom.

        Args:
            u_nom: nominal ego acceleration.
            h: current rear gap h_r = s_ego - s_rear - L.
            v_ego, v_rear: current speeds.
        """
        self.last_h = h - self.d_min
        lb = self.lower_bound(h, v_ego, v_rear)
        lo = max(-self.a_e, lb)                 # combine HOCBF bound with brake limit
        # Closest-to-nominal subject to  lo <= a <= a_acc  (a_acc wins if infeasible).
        a = min(max(u_nom, lo), self.a_acc)
        self.last_intervened = abs(a - u_nom) > 1e-3
        return float(a)
