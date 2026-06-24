"""
Plain CBF-QP safety filter for 1D car following (relative degree 1).

Time-headway barrier:
    h  = (s_lead - L) - s_ego - d_min - tau * v_ego
    h_dot = (v_lead - v_ego) - tau * u
The tau*v_ego term brings the control u into h_dot, so h has relative degree 1
and a single CBF constraint suffices:
    h_dot + alpha * h >= 0   =>   tau * u <= (v_lead - v_ego) + alpha * h

QP:  minimize (u - u_nom)^2  s.t. the constraint above and |u| <= a_max.
"""

import numpy as np
import cvxpy as cp


class CarFollowingCBF1D:
    """Relative-degree-1 time-headway CBF-QP filter for the 1D ego car."""

    def __init__(self, robot_spec: dict, alpha: float = 1.0,
                 tau: float = 1.0, d_min: float = 2.0):
        """
        Args:
            robot_spec: must contain 'a_max' (and optionally 'body_length' L).
            alpha: class-K gain on the barrier.
            tau: time headway [s].
            d_min: minimum bumper-to-bumper standstill gap [m].
        """
        self.a_max = float(robot_spec.get('a_max', 3.0))
        self.L = float(robot_spec.get('body_length', 4.5))
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.d_min = float(d_min)
        self.last_h = None

    def barrier(self, x_ego, s_lead: float, v_lead: float) -> float:
        """Headway barrier value h (>= 0 means safe)."""
        x = np.array(x_ego).flatten()
        s_ego, v_ego = float(x[0]), float(x[1])
        return (s_lead - self.L) - s_ego - self.d_min - self.tau * v_ego

    def filter(self, u_nom: float, x_ego, s_lead: float, v_lead: float) -> float:
        """Return the safe control closest to u_nom.

        Args:
            u_nom: nominal acceleration.
            x_ego: ego state [s, v].
            s_lead, v_lead: current lead-car position (center) and speed.

        Returns:
            Safe acceleration in [-a_max, a_max].
        """
        x = np.array(x_ego).flatten()
        v_ego = float(x[1])

        h = self.barrier(x_ego, s_lead, v_lead)
        self.last_h = h

        # tau * u <= (v_lead - v_ego) + alpha * h
        rhs = (v_lead - v_ego) + self.alpha * h

        u = cp.Variable()
        objective = cp.Minimize((u - u_nom) ** 2)
        constraints = [
            self.tau * u <= rhs,
            u >= -self.a_max,
            u <= self.a_max,
        ]
        prob = cp.Problem(objective, constraints)

        status = 'failure'
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            status = prob.status
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                status = prob.status
            except Exception:
                status = 'failure'

        if status in ('optimal', 'optimal_inaccurate') and u.value is not None:
            return float(np.clip(u.value, -self.a_max, self.a_max))
        # Infeasible / solver failure -> brake hard.
        return -self.a_max
