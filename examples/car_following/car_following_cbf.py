"""
Exact CBF-QP safety filter for 1D car following.

Uses the maximal invariant set of He & Orosz (ITSC 2018) as the barrier:
    b(h, v, v1) = h - b_hat(v, v1)        (paper eq. 23)
with b_hat the exact (piecewise) safe-distance surface; see
car_following_safe_distance.py.

CBF condition  b_dot + gamma * b >= 0,  with  h_dot = v1 - v,  v_dot = a (control),
v1_dot = a1 (lead acceleration), taken at the worst case a1 = -a_l:

    b_dot = (v1 - v) - (db_hat/dv) * a - (db_hat/dv1) * a1
          = (v1 - v) - (db_hat/dv) * a + (db_hat/dv1) * a_l     [a1 = -a_l]

Since db_hat/dv > 0 (paper eq. 27) the constraint is an upper bound on a:

    a <= [ gamma * b + (v1 - v) + (db_hat/dv1) * a_l ] / (db_hat/dv)

QP:  minimize (a - a_nom)^2  s.t. that bound and  a in [-a_e, a_bar].
"""

import numpy as np
import cvxpy as cp

from car_following_safe_distance import safe_distance, safe_distance_grad


class CarFollowingCBF1D:
    """Exact-CBF QP filter using the maximal-invariant-set barrier b = h - b_hat."""

    def __init__(self, robot_spec: dict, p: dict, gamma: float = 1.0,
                 L: float = 4.5):
        """
        Args:
            robot_spec: must contain 'a_max' (= a_bar, max acceleration magnitude).
            p: safe-distance params dict (tau, a_e, a_l, vbar). a_e is the ego
               max deceleration used by the QP control bound and the barrier.
            gamma: class-K gain (extended class-K pi(b) = gamma * b).
            L: combined center-to-bumper offset (sum of half-lengths).
        """
        self.p = dict(p)
        self.a_e = float(p['a_e'])
        self.a_l = float(p['a_l'])
        self.a_bar = float(robot_spec.get('a_max', 3.0))
        self.gamma = float(gamma)
        self.L = float(L)
        self.last_h = None          # last barrier value b
        self.last_status = None

    def headway(self, x_ego, s_lead: float) -> float:
        """Bumper-to-bumper distance headway h."""
        s_ego = float(np.array(x_ego).flatten()[0])
        return (s_lead - self.L) - s_ego

    def barrier(self, x_ego, s_lead: float, v_lead: float) -> float:
        """CBF value b = h - b_hat(v, v1)  (>= 0 means inside the invariant set)."""
        x = np.array(x_ego).flatten()
        v_ego = float(x[1])
        h = self.headway(x_ego, s_lead)
        return h - safe_distance(v_ego, v_lead, self.p)

    def accel_upper_bound(self, x_ego, s_lead: float, v_lead: float) -> float:
        """Forward CBF upper bound on the ego acceleration: the constraint is `a <= rhs`.

        This is the raw CBF condition (worst-case lead decel `a1 = -a_l`) before the QP /
        actuator clamp, exposed so a combined controller (e.g. the sandwiched front+rear
        filter) can pair it with a separate lower bound. Also sets `last_h` (barrier b).
        """
        x = np.array(x_ego).flatten()
        v_ego = float(x[1])

        b = self.barrier(x_ego, s_lead, v_lead)
        self.last_h = b

        db_dv, db_dv1 = safe_distance_grad(v_ego, v_lead, self.p)
        db_dv = max(db_dv, 1e-6)        # > 0 by construction; guard division

        # a <= rhs  (worst-case a1 = -a_l)
        return (self.gamma * b + (v_lead - v_ego) + db_dv1 * self.a_l) / db_dv

    def filter(self, u_nom: float, x_ego, s_lead: float, v_lead: float) -> float:
        """Return the safe acceleration closest to u_nom.

        Args:
            u_nom: nominal acceleration.
            x_ego: ego state [s, v].
            s_lead, v_lead: current lead position (center) and speed (= v1).
        """
        rhs = self.accel_upper_bound(x_ego, s_lead, v_lead)

        a = cp.Variable()
        objective = cp.Minimize((a - u_nom) ** 2)
        constraints = [a <= rhs, a >= -self.a_e, a <= self.a_bar]
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
        self.last_status = status

        if status in ('optimal', 'optimal_inaccurate') and a.value is not None:
            return float(np.clip(a.value, -self.a_e, self.a_bar))
        # Infeasible / solver failure -> brake hard.
        return -self.a_e
