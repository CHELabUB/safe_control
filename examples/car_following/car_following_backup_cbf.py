"""
Backup CBF for 1D car following behind a (worst-case braking) lead car.

Specializes BackupCBF1D (the rollout backup filter) for the 3-state car-following
problem [h, v, v1] of He & Orosz (ITSC 2018), with the *smooth* invariant surface
b_hat_s(v, v1) >= b_hat as the terminal set:

  - Backup policy: ego brakes at -a_e to a stop and holds (never reverses).
  - Lead prediction over the horizon: worst-case constant deceleration -a_l to a
    stop, supplied via set_moving_obstacles(callable) by the run script.
  - Safety along the rollout: h(t) - d_min >= 0 (no collision while braking).
  - Terminal set: h(T) - b_hat_s(v(T), v1(T)) >= 0, i.e. the ego brakes into the
    smooth maximal-invariant set, where it is safe thereafter even if the lead
    keeps braking.  b_hat_s is differentiable, so the terminal QP gradient is clean.

Only the geometry (barriers + brake-to-stop flow) is specialized; the scalar QP
and state-transition-matrix scaffolding are reused from BackupCBF1D / BackupCBF.
"""

import os
import sys

import numpy as np
import cvxpy as cp

# Reach BackupCBF1D / DoubleIntegrator1D in the sibling example dir.
_DI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'double_integrator')
sys.path.insert(0, os.path.abspath(_DI_DIR))
from backup_cbf_1d_wrapper import BackupCBF1D  # noqa: E402

from car_following_safe_distance import SmoothSafeDistance  # noqa: E402


class CarFollowingBackupCBF1D(BackupCBF1D):
    """Rollout backup CBF that brakes the ego into the smooth invariant set."""

    def __init__(self, robot, robot_spec, p, dt=0.05, backup_horizon=4.0,
                 L=4.5, d_min=2.0, gamma=1.0, gamma_terminal=1.0,
                 smooth=None, ax=None):
        """
        Args:
            robot: DoubleIntegrator1D instance.
            robot_spec: must contain 'u_max' (set to a_e for QP scaling/bounds).
            p: safe-distance params (tau, a_e, a_l, vbar).
            dt, backup_horizon: discretization and horizon T.
            L: combined center-to-bumper offset.
            d_min: minimum bumper-to-bumper standstill gap (rollout collision margin).
            gamma, gamma_terminal: class-K gains for safety / terminal constraints.
            smooth: optional prebuilt SmoothSafeDistance (else built from p).
        """
        self.p = dict(p)
        self.a_e = float(p['a_e'])
        self.a_l = float(p['a_l'])
        super().__init__(
            robot, robot_spec, dt=dt, backup_horizon=backup_horizon,
            u_b=-self.a_e,                              # backup = max brake
            alpha_fn=(lambda h, g=gamma: g * h),
            alpha_terminal_fn=(lambda h, g=gamma_terminal: g * h),
            eps_s=0.0, eps_v=0.0, ax=ax,
        )
        self.a_max = self.a_e                           # brake-to-stop flow uses a_e
        self.L = float(L)
        self.d_min = float(d_min)
        self.smooth = smooth if smooth is not None else SmoothSafeDistance(self.p)

    # ------------------------------------------------------------------
    # Predicted lead state from the moving-obstacle callable
    # ------------------------------------------------------------------

    def _lead(self, t: float):
        obs = self._get_obstacle_at_time(t)
        if obs is None:
            return float('inf'), 0.0
        return float(obs['x']), float(obs.get('vx', 0.0))

    # ------------------------------------------------------------------
    # Safety CBF along the rollout: no collision while braking
    # ------------------------------------------------------------------

    def _h_safety(self, x, t: float = 0.0) -> float:
        s_ego = float(np.array(x).flatten()[0])
        s_lead, _ = self._lead(t)
        return (s_lead - self.L) - s_ego - self.d_min

    def _grad_h_safety(self, x, t: float = 0.0) -> np.ndarray:
        return np.array([-1.0, 0.0])

    # ------------------------------------------------------------------
    # Terminal set: ego braked into the smooth invariant set
    # ------------------------------------------------------------------

    def _h_terminal_t(self, x, t: float) -> float:
        xf = np.array(x).flatten()
        s_ego, v_ego = float(xf[0]), float(xf[1])
        s_lead, v_lead = self._lead(t)
        h = (s_lead - self.L) - s_ego
        return h - self.smooth.value(v_ego, v_lead)

    def _h_terminal(self, x) -> float:
        return self._h_terminal_t(x, self.backup_horizon)

    def _grad_h_terminal(self, x) -> np.ndarray:
        xf = np.array(x).flatten()
        s_lead, v_lead = self._lead(self.backup_horizon)
        db_dv, _ = self.smooth.grad(float(xf[1]), v_lead)
        return np.array([-1.0, -db_dv])      # d/d[s_ego, v_ego]

    # ------------------------------------------------------------------
    # Backup flow: brake at a_e to a stop, then hold (analytical)
    # ------------------------------------------------------------------

    def _integrate_backup_trajectory_analytical(self, x0) -> tuple:
        """Brake-to-stop-then-hold flow map and sensitivity (STM)."""
        x0f = np.array(x0).flatten()
        s0, v0 = float(x0f[0]), float(x0f[1])
        a = self.a_e

        t = np.arange(self.N) * self.dt
        v0p = max(v0, 0.0)
        t_stop = v0p / a if a > 0 else 0.0
        s_final = s0 + (v0p ** 2) / (2.0 * a) if a > 0 else s0

        moving = t <= t_stop

        phi = np.zeros((self.N, 2))
        phi[moving, 0] = s0 + v0 * t[moving] - 0.5 * a * t[moving] ** 2
        phi[moving, 1] = v0 - a * t[moving]
        phi[~moving, 0] = s_final
        phi[~moving, 1] = 0.0

        S = np.zeros((self.N, 2, 2))
        S[moving, 0, 0] = 1.0
        S[moving, 0, 1] = t[moving]
        S[moving, 1, 1] = 1.0
        S[~moving, 0, 0] = 1.0
        S[~moving, 0, 1] = t_stop          # d s_final / d v0 = v0p / a = t_stop

        return phi, S

    # ------------------------------------------------------------------
    # QP: BackupCBF1D scalar QP + moving-lead dh/dt term (safety & terminal)
    # ------------------------------------------------------------------

    def solve_control_problem(self, robot_state, friction=None):
        robot_state = np.array(robot_state).flatten()

        phi, S = self._integrate_backup_trajectory(robot_state)

        h_vals = [self._h_safety(phi[i], i * self.dt) for i in range(self.N)]
        h_term = self._h_terminal(phi[-1])
        self._last_h_min = min(float(np.min(h_vals)), h_term)
        if self._last_h_min < self.global_min_h:
            self.global_min_h = self._last_h_min

        if self.visualize_backup and self.curr_step % self.save_every_N == 0:
            self.backup_trajs.append(phi.copy())
        self.latest_backup_trajectory = phi.copy()
        self.curr_step += 1

        u_nom = self._get_nominal_control(robot_state)
        u_max = self.robot_spec.get('u_max', 1.0)
        u_nom = float(np.clip(u_nom, -u_max, u_max).flat[0])

        f0 = self._dynamics_f(robot_state)
        g0 = self._dynamics_g(robot_state)

        coeffs = []
        rhs_vals = []

        for i in range(1, self.N):
            x_i = phi[i]
            S_i = S[i]
            t_i = i * self.dt

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

        # Terminal constraint with moving-lead dh/dt term.
        x_T, S_T = phi[-1], S[-1]
        t_T = (self.N - 1) * self.dt
        h_terminal = self._h_terminal_t(x_T, t_T)
        grad_hT = self._grad_h_terminal(x_T)
        dhT_dt = (self._h_terminal_t(x_T, t_T + self.dt) - h_terminal) / self.dt
        c_T = float((grad_hT @ S_T @ g0).flat[0])
        r_T = float(-(grad_hT @ S_T @ f0) - dhT_dt
                    - self._alpha_terminal(h_terminal))
        has_terminal = abs(c_T) > 1e-6

        # Soft-constrained QP: slacks make it always feasible and graceful near
        # the backup's rest point, where the terminal STM column degenerates
        # (c_T -> 0).  Safety slacks carry a much larger penalty than the terminal
        # slack, so collision-avoidance dominates terminal-set shaping.
        RHO_SAFE, RHO_TERM = 1e4, 1e1
        status = 'no_constraints'
        if len(coeffs) > 0 or has_terminal:
            u_s = cp.Variable()
            objective = (u_s - u_nom / u_max) ** 2
            constraints = [u_s >= -1.0, u_s <= 1.0]

            if len(coeffs) > 0:
                coeffs_arr = np.array(coeffs)
                rhs_arr = np.array(rhs_vals)
                xi_s = cp.Variable(len(coeffs), nonneg=True)
                constraints.append((coeffs_arr * u_max) * u_s + xi_s >= rhs_arr)
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

            # The soft-constrained QP is always feasible, so any returned point is
            # usable. Accept it even for 'user_limit'/'*_inaccurate' (OSQP's best
            # iterate); only fall back to the hard brake on a genuine failure
            # (no solution). Slamming -a_e on every non-converged step would make
            # the control chatter between the QP value and -a_e.
            bad = ('infeasible', 'infeasible_inaccurate', 'unbounded', 'failure')
            if u_s.value is not None and status not in bad:
                u_safe = float(np.clip(u_s.value, -1.0, 1.0)) * u_max
                self._last_intervention = abs(u_safe - u_nom) > 0.01 * u_max
                self._using_backup = self._last_intervention
            else:
                u_safe = float(self.u_b)            # backup brake (-a_e)
                self._last_intervention = True
                self._using_backup = True
        else:
            u_safe = u_nom
            self._last_intervention = False
            self._using_backup = False

        self.last_status = status
        self._update_visualization(phi)
        return np.array([[u_safe]])
