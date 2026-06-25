"""
Rear-end-avoidance Backup CBF for the ego (1D).

Mirror image of the forward CarFollowingBackupCBF1D: instead of braking to a stop
behind a lead, the ego *accelerates to v_max* to escape an unfiltered follower
behind it.

  - Backup policy: u = +u_max, accelerate to v_max then hold (analytic flow; the
    state-transition matrix mirrors brake-to-stop with t_reach in place of t_stop).
  - Obstacle = the REAR vehicle, predicted by simulating its *assumed* car-following
    model (OVM/IDM) against the ego's backup trajectory (the rear reacts to the ego).
  - Barrier:  h_r(x, t) = (s_ego - L) - s_rear_pred(t) - d_min   (grad wrt ego = [+1, 0]).
  - Terminal set: ego has reached v_max with h_r >= d_min (then h_r is non-decreasing
    since the rear is capped at v_max, so the state is safe thereafter).

Sensitivity of the predicted h_r to the current ego state (used in the QP):
  - 'analytic' (default): ego analytic STM with the rear prediction treated as an
    exogenous moving obstacle (its motion enters via the explicit dh/dt term); the
    receding horizon refreshes the prediction each step.
  - 'coupled': finite-difference the *joint* ego+rear rollout, capturing the rear's
    reaction to a perturbed ego state exactly.

State x = [s, v], control u = acceleration, g(x) = [0, 1]^T.
"""

import os
import sys

import numpy as np
import cvxpy as cp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'double_integrator')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'car_following')))

from backup_cbf_1d_wrapper import BackupCBF1D            # noqa: E402
from car_following_models import nominal_accel           # noqa: E402

BODY_LENGTH = 4.5


class RearEndBackupCBF1D(BackupCBF1D):
    """Backup CBF that accelerates the ego to v_max to avoid being rear-ended."""

    def __init__(self, robot, robot_spec, dt=0.05, backup_horizon=5.0,
                 u_acc=3.0, v_max=12.0, L=4.5, d_min=1.0,
                 rear_model='ovm', rear_params=None, sensitivity='analytic',
                 gamma=1.0, gamma_terminal=2.0, eps_v=0.1, ax=None):
        """
        Args:
            robot: DoubleIntegrator1D instance.
            robot_spec: must contain 'u_max' (= u_acc for QP scaling/bounds).
            backup_horizon: horizon T; should exceed (v_max - 0)/u_acc to let the
                ego actually reach v_max within the rollout.
            u_acc: backup acceleration magnitude (escape).
            v_max: speed the ego accelerates to.
            L, d_min: combined length and rear collision margin.
            rear_model, rear_params: the ego's *assumed* model of the rear (may
                differ from the rear's true params).
            sensitivity: 'analytic' or 'coupled'.
        """
        super().__init__(
            robot, robot_spec, dt=dt, backup_horizon=backup_horizon,
            u_b=+abs(u_acc),
            alpha_fn=(lambda h, g=gamma: g * h),
            alpha_terminal_fn=(lambda h, g=gamma_terminal: g * h),
            eps_s=0.0, eps_v=eps_v, ax=ax,
        )
        self.u_acc = float(abs(u_acc))
        self.v_max = float(v_max)
        self.L = float(L)
        self.d_min = float(d_min)
        self.rear_model = rear_model
        self.rear_params = dict(rear_params or {})
        self.sensitivity = sensitivity
        self._rear0 = (0.0, 0.0)           # current rear state (s_r, v_r)
        self._rear_s = None                # predicted rear positions over horizon
        self._rear_v = None
        self.last_status = 'none'

    def set_rear_state(self, s_rear: float, v_rear: float):
        """Provide the rear vehicle's current state before solve_control_problem."""
        self._rear0 = (float(s_rear), float(v_rear))

    # ------------------------------------------------------------------
    # Ego backup flow: accelerate to v_max then hold (analytic)
    # ------------------------------------------------------------------

    def _ego_flow(self, s0, v0):
        a = self.u_acc
        t = np.arange(self.N) * self.dt
        v0 = float(v0)
        t_reach = (self.v_max - v0) / a if (a > 0 and v0 < self.v_max) else 0.0
        accel = t <= t_reach
        s = np.empty(self.N)
        v = np.empty(self.N)
        v[accel] = v0 + a * t[accel]
        s[accel] = s0 + v0 * t[accel] + 0.5 * a * t[accel] ** 2
        s_reach = s0 + v0 * t_reach + 0.5 * a * t_reach ** 2
        v[~accel] = self.v_max
        s[~accel] = s_reach + self.v_max * (t[~accel] - t_reach)
        return s, v, t_reach

    def _integrate_backup_trajectory_analytical(self, x0):
        x0f = np.array(x0).flatten()
        s0, v0 = float(x0f[0]), float(x0f[1])
        s, v, t_reach = self._ego_flow(s0, v0)
        t = np.arange(self.N) * self.dt
        accel = t <= t_reach

        phi = np.stack([s, v], axis=1)
        S = np.zeros((self.N, 2, 2))
        S[accel, 0, 0] = 1.0
        S[accel, 0, 1] = t[accel]
        S[accel, 1, 1] = 1.0
        S[~accel, 0, 0] = 1.0
        S[~accel, 0, 1] = t_reach          # d s / d v0 after reaching v_max
        return phi, S

    # ------------------------------------------------------------------
    # Rear prediction: simulate the assumed rear model against the ego flow
    # ------------------------------------------------------------------

    def _rollout_rear(self, ego_s, ego_v):
        s_r0, v_r0 = self._rear0
        s_r = np.empty(self.N)
        v_r = np.empty(self.N)
        s_r[0], v_r[0] = s_r0, v_r0
        for i in range(self.N - 1):
            ego_d = {'x': ego_s[i], 'vx': ego_v[i], 'length': BODY_LENGTH}
            rear_d = {'x': s_r[i], 'vx': v_r[i], 'length': BODY_LENGTH}
            a = nominal_accel(self.rear_model, rear_d, ego_d, self.rear_params)
            a = max(a, -v_r[i] / self.dt)          # no reversing
            v_r[i + 1] = max(v_r[i] + a * self.dt, 0.0)
            s_r[i + 1] = s_r[i] + v_r[i] * self.dt
        return s_r, v_r

    def _rear_s_at(self, t):
        i = int(round(t / self.dt))
        i = min(max(i, 0), self.N - 1)
        return self._rear_s[i]

    # ------------------------------------------------------------------
    # Barriers (use the stored rear prediction self._rear_s)
    # ------------------------------------------------------------------

    def _h_safety(self, x, t=0.0):
        s_ego = float(np.array(x).flatten()[0])
        return (s_ego - self.L) - self._rear_s_at(t) - self.d_min

    def _grad_h_safety(self, x, t=0.0):
        return np.array([1.0, 0.0])

    def _h_terminal_t(self, x, t):
        # Gap at the horizon. By construction the ego is at v_max there (T exceeds
        # the time to reach v_max), and the rear is capped at v_max, so h_r is
        # non-decreasing beyond T -> a positive gap is invariant. A rigid "must be
        # at v_max" term is intentionally omitted: it would fight the braking
        # nominal once the rear is far away and make the control chatter.
        s_ego = float(np.array(x).flatten()[0])
        return (s_ego - self.L) - self._rear_s_at(t) - self.d_min

    def _h_terminal(self, x):
        return self._h_terminal_t(x, (self.N - 1) * self.dt)

    def _grad_h_terminal(self, x):
        return np.array([1.0, 0.0])

    # ------------------------------------------------------------------
    # Coupled finite-difference: h_r(t_i) as a function of ego (s0, v0)
    # ------------------------------------------------------------------

    def _joint_h(self, s0, v0):
        """h_r over the horizon for ego initial (s0, v0), joint ego+rear rollout."""
        ego_s, ego_v, _ = self._ego_flow(s0, v0)
        s_r, _ = self._rollout_rear(ego_s, ego_v)
        return (ego_s - self.L) - s_r - self.d_min

    # ------------------------------------------------------------------
    # QP (soft-constrained; mirrors CarFollowingBackupCBF1D)
    # ------------------------------------------------------------------

    def solve_control_problem(self, robot_state, friction=None):
        x0 = np.array(robot_state).flatten()
        s0, v0 = float(x0[0]), float(x0[1])

        phi, S = self._integrate_backup_trajectory(x0)
        # Predict the rear against the ego backup trajectory; store for barriers.
        self._rear_s, self._rear_v = self._rollout_rear(phi[:, 0], phi[:, 1])

        h_vals = [self._h_safety(phi[i], i * self.dt) for i in range(self.N)]
        self._last_h_min = min(float(np.min(h_vals)), self._h_terminal(phi[-1]))
        self.latest_backup_trajectory = phi.copy()
        self.curr_step += 1

        u_nom = self._get_nominal_control(x0)
        u_max = self.robot_spec.get('u_max', 1.0)
        u_nom = float(np.clip(u_nom, -u_max, u_max).flat[0])

        f0 = self._dynamics_f(x0)
        g0 = self._dynamics_g(x0)

        coeffs, rhs_vals = [], []

        if self.sensitivity == 'coupled':
            # Same constraint as 'analytic', but the sensitivity of h_r(t_i) to the
            # current ego state is the *joint* finite difference (captures the rear
            # reacting to a perturbed ego) instead of the ego-only STM.
            d = 1e-3
            h0 = self._joint_h(s0, v0)
            dh_ds = (self._joint_h(s0 + d, v0) - h0) / d
            dh_dv = (self._joint_h(s0, v0 + d) - h0) / d
            for i in range(1, self.N):
                if i < self.N - 1:
                    f_pi = (phi[i + 1] - phi[i]) / self.dt
                else:
                    f_pi = (phi[i] - phi[i - 1]) / self.dt
                v_ego_i = float(f_pi[0])                       # grad_h_local . f_pi
                dh_dt = (self._h_safety(phi[i], i * self.dt + self.dt)
                         - self._h_safety(phi[i], i * self.dt)) / self.dt
                c = float(dh_dv[i])
                r = float(-dh_ds[i] * v0 + v_ego_i - dh_dt - self._alpha(h0[i]))
                if abs(c) > 1e-6:
                    coeffs.append(c)
                    rhs_vals.append(r)
            # terminal: gap at horizon (ego is already at v_max there by construction)
            hT = float(h0[-1])
            c_T = float(dh_dv[-1])
            dhT_dt = (self._h_safety(phi[-1], (self.N - 1) * self.dt + self.dt)
                      - self._h_safety(phi[-1], (self.N - 1) * self.dt)) / self.dt
            r_T = float(-dh_ds[-1] * v0 - dhT_dt - self._alpha_terminal(hT))
            has_terminal = abs(c_T) > 1e-6
        else:
            # Analytic: ego STM, rear exogenous (its motion via explicit dh/dt).
            for i in range(1, self.N):
                x_i, S_i, t_i = phi[i], S[i], i * self.dt
                h_val = self._h_safety(x_i, t_i)
                grad_h = self._grad_h_safety(x_i, t_i)
                dh_dt = (self._h_safety(x_i, t_i + self.dt) - h_val) / self.dt
                if i < self.N - 1:
                    f_pi = (phi[i + 1] - phi[i]) / self.dt
                else:
                    f_pi = (phi[i] - phi[i - 1]) / self.dt
                c = float((grad_h @ S_i @ g0).flat[0])
                r = float(-(grad_h @ S_i @ f0) + (grad_h @ f_pi) - dh_dt
                          - self._alpha(h_val))
                if abs(c) > 1e-6:
                    coeffs.append(c)
                    rhs_vals.append(r)
            x_T, S_T, t_T = phi[-1], S[-1], (self.N - 1) * self.dt
            hT = self._h_terminal_t(x_T, t_T)
            grad_hT = self._grad_h_terminal(x_T)
            dhT_dt = (self._h_terminal_t(x_T, t_T + self.dt) - hT) / self.dt
            c_T = float((grad_hT @ S_T @ g0).flat[0])
            r_T = float(-(grad_hT @ S_T @ f0) - dhT_dt - self._alpha_terminal(hT))
            has_terminal = abs(c_T) > 1e-6

        RHO_SAFE, RHO_TERM = 1e4, 1e1
        status = 'no_constraints'
        if coeffs or has_terminal:
            u_s = cp.Variable()
            objective = (u_s - u_nom / u_max) ** 2
            constraints = [u_s >= -1.0, u_s <= 1.0]
            if coeffs:
                ca = np.array(coeffs)
                ra = np.array(rhs_vals)
                xi_s = cp.Variable(len(coeffs), nonneg=True)
                constraints.append((ca * u_max) * u_s + xi_s >= ra)
                objective = objective + RHO_SAFE * cp.sum_squares(xi_s)
            if has_terminal:
                xi_t = cp.Variable(nonneg=True)
                constraints.append(c_T * u_max * u_s + xi_t >= r_T)
                objective = objective + RHO_TERM * xi_t ** 2
            prob = cp.Problem(cp.Minimize(objective), constraints)
            status = 'failure'
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                           max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
                status = prob.status
            except Exception:
                try:
                    prob.solve(solver=cp.SCS, verbose=False)
                    status = prob.status
                except Exception:
                    status = 'failure'

            bad = ('infeasible', 'infeasible_inaccurate', 'unbounded', 'failure')
            if u_s.value is not None and status not in bad:
                u_safe = float(np.clip(u_s.value, -1.0, 1.0)) * u_max
            else:
                u_safe = float(self.u_b)        # escape: accelerate (+u_acc)
        else:
            u_safe = u_nom

        self.last_status = status
        return np.array([[u_safe]])
