"""
Backup CBF for 1D car following behind a moving lead car.

Specializes BackupCBF1D (the fixed-wall double-integrator backup filter) for a
*moving* obstacle:

  - Backup policy: max braking then hold (ego brakes at a_max to a stop and
    never reverses).  This replaces the wall demo's "constant positive u_b".
  - Safety barrier: h(x, t) = (s_lead_pred(t) - L) - s_ego - d_min, where the
    predicted lead position over the backup horizon is supplied through
    set_moving_obstacles(callable).
  - Terminal set: ego stopped a safe distance behind the lead.
  - The QP adds the moving-obstacle time-derivative term dh/dt (= lead speed),
    which BackupCBF1D.solve_control_problem omits but the base BackupCBF includes.

State x = [s, v], control u = acceleration, g(x) = [0, 1]^T.
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


class CarFollowingBackupCBF1D(BackupCBF1D):
    """Backup CBF that brakes-to-stop safely behind a moving lead car."""

    def __init__(self, robot, robot_spec, dt=0.02, backup_horizon=2.0,
                 a_max=3.0, L=4.5, d_min=2.0,
                 alpha_fn=None, alpha_terminal_fn=None,
                 eps_s=0.05, eps_v=0.05, ax=None):
        """
        Args:
            robot: DoubleIntegrator1D instance.
            robot_spec: must contain 'u_max' (= a_max for QP scaling/bounds).
            dt, backup_horizon: discretization and horizon T.
            a_max: maximum braking magnitude (backup decelerates at a_max).
            L: combined center-to-bumper offset (sum of ego/lead half-lengths).
            d_min: minimum bumper-to-bumper standstill gap.
            alpha_fn, alpha_terminal_fn: class-K functions (default identity).
            eps_s, eps_v: terminal-set margins.
        """
        # Backup control = brake at -a_max; also the emergency fallback value.
        super().__init__(
            robot, robot_spec, dt=dt, backup_horizon=backup_horizon,
            u_b=-abs(a_max), alpha_fn=alpha_fn,
            alpha_terminal_fn=alpha_terminal_fn,
            eps_s=eps_s, eps_v=eps_v, ax=ax,
        )
        self.a_max = float(abs(a_max))
        self.L = float(L)
        self.d_min = float(d_min)

    # ------------------------------------------------------------------
    # Predicted lead position from the moving-obstacle callable
    # ------------------------------------------------------------------

    def _lead_pos(self, t: float) -> float:
        """Predicted lead-car center position at backup-relative time t."""
        obs = self._get_obstacle_at_time(t)
        if obs is None:
            return float('inf')   # no obstacle set -> trivially safe
        return float(obs['x'])

    # ------------------------------------------------------------------
    # Safety CBF: moving lead, relative degree handled by backup rollout
    # ------------------------------------------------------------------

    def _h_safety(self, x, t: float = 0.0) -> float:
        s_ego = float(np.array(x).flatten()[0])
        return (self._lead_pos(t) - self.L) - s_ego - self.d_min

    def _grad_h_safety(self, x, t: float = 0.0) -> np.ndarray:
        return np.array([-1.0, 0.0])

    # ------------------------------------------------------------------
    # Terminal set: ego stopped, safe gap behind the (predicted) lead at T
    # ------------------------------------------------------------------

    def _h_terminal(self, x) -> float:
        xf = np.array(x).flatten()
        h_gap = self._h_safety(xf, self.backup_horizon) - self.eps_s
        h_stop = self.eps_v - xf[1]            # want v -> 0 (stopped, no reverse drive)
        return float(min(h_gap, h_stop))

    def _grad_h_terminal(self, x) -> np.ndarray:
        xf = np.array(x).flatten()
        h_gap = self._h_safety(xf, self.backup_horizon) - self.eps_s
        h_stop = self.eps_v - xf[1]
        if h_gap <= h_stop:
            return np.array([-1.0, 0.0])       # gap term active: dh/ds_ego = -1
        return np.array([0.0, -1.0])           # stop term active: dh/dv = -1

    # ------------------------------------------------------------------
    # Backup flow: brake at a_max to a stop, then hold (analytical)
    # ------------------------------------------------------------------

    def _integrate_backup_trajectory_analytical(self, x0) -> tuple:
        """Brake-to-stop-then-hold flow map and sensitivity (STM).

        Moving phase (t <= t_stop = max(v0,0)/a):
            s = s0 + v0 t - 1/2 a t^2,   v = v0 - a t,   S = [[1, t], [0, 1]]
        Stopped phase (t > t_stop):
            s = s0 + v0p^2/(2a) (held),  v = 0,          S = [[1, t_stop], [0, 0]]
        where v0p = max(v0, 0).  The velocity row is zeroed post-stop because
        v == 0 there regardless of x0, and d s_final / d v0 = v0p/a = t_stop.
        """
        x0f = np.array(x0).flatten()
        s0, v0 = float(x0f[0]), float(x0f[1])
        a = self.a_max

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
        # Moving phase STM [[1, t], [0, 1]]
        S[moving, 0, 0] = 1.0
        S[moving, 0, 1] = t[moving]
        S[moving, 1, 1] = 1.0
        # Stopped phase STM [[1, t_stop], [0, 0]]
        S[~moving, 0, 0] = 1.0
        S[~moving, 0, 1] = t_stop

        return phi, S

    # ------------------------------------------------------------------
    # QP: BackupCBF1D scalar QP + moving-obstacle dh/dt term
    # ------------------------------------------------------------------

    def solve_control_problem(self, robot_state, friction=None):
        """Backup CBF-QP for the moving-lead car-following problem.

        Identical to BackupCBF1D.solve_control_problem except each safety
        constraint includes dh/dt (the lead's contribution to h_dot), evaluated
        by finite difference of the time-dependent barrier.
        """
        robot_state = np.array(robot_state).flatten()

        phi, S = self._integrate_backup_trajectory(robot_state)

        # Track safety metrics
        h_vals = [self._h_safety(phi[i], i * self.dt) for i in range(self.N)]
        h_term = self._h_terminal(phi[-1])
        self._last_h_min = min(float(np.min(h_vals)), h_term)
        if self._last_h_min < self.global_min_h:
            self.global_min_h = self._last_h_min

        if self.visualize_backup and self.curr_step % self.save_every_N == 0:
            self.backup_trajs.append(phi.copy())
        self.latest_backup_trajectory = phi.copy()
        self.curr_step += 1

        # Nominal control (scalar)
        u_nom = self._get_nominal_control(robot_state)
        u_max = self.robot_spec.get('u_max', 1.0)
        u_nom = float(np.clip(u_nom, -u_max, u_max).flat[0])

        f0 = self._dynamics_f(robot_state)   # (2,)
        g0 = self._dynamics_g(robot_state)   # (2, 1)

        coeffs = []
        rhs_vals = []

        for i in range(1, self.N):
            x_i = phi[i]
            S_i = S[i]
            t_i = i * self.dt

            h_val = self._h_safety(x_i, t_i)
            grad_h = self._grad_h_safety(x_i, t_i)   # (2,)

            # Moving-obstacle time derivative dh/dt (= lead speed contribution).
            dh_dt = (self._h_safety(x_i, t_i + self.dt) - h_val) / self.dt

            # Backup drift: d/dt phi at time t_i
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

        # Terminal constraint (fixed horizon; no dh/dt term, matching base CBF)
        x_T, S_T = phi[-1], S[-1]
        h_terminal = self._h_terminal(x_T)
        grad_hT = self._grad_h_terminal(x_T)
        c_T = float((grad_hT @ S_T @ g0).flat[0])
        r_T = float(-(grad_hT @ S_T @ f0 + self._alpha_terminal(h_terminal)))

        if abs(c_T) > 1e-6:
            coeffs.append(c_T)
            rhs_vals.append(r_T)

        if len(coeffs) > 0:
            coeffs_arr = np.array(coeffs)
            rhs_arr = np.array(rhs_vals)

            u_s = cp.Variable()
            objective = cp.Minimize((u_s - u_nom / u_max) ** 2)
            constraints = [
                coeffs_arr * u_max * u_s >= rhs_arr,
                u_s >= -1.0,
                u_s <= 1.0,
            ]
            prob = cp.Problem(objective, constraints)

            prob_status = 'failure'
            if not (np.any(np.isnan(coeffs_arr)) or np.any(np.isnan(rhs_arr))):
                try:
                    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                    prob_status = prob.status
                except Exception:
                    try:
                        prob.solve(solver=cp.SCS, verbose=False)
                        prob_status = prob.status
                    except Exception:
                        prob_status = 'failure'

            if prob_status in ['optimal', 'optimal_inaccurate'] and u_s.value is not None:
                u_safe = float(u_s.value) * u_max
                self._last_intervention = abs(u_safe - u_nom) > 0.01 * u_max
                self._using_backup = self._last_intervention
            else:
                if self._last_h_min > 0.01:
                    u_safe = u_nom
                    self._last_intervention = False
                    self._using_backup = False
                else:
                    u_safe = float(self.u_b)   # emergency brake (-a_max)
                    self._last_intervention = True
                    self._using_backup = True
        else:
            u_safe = u_nom
            self._last_intervention = False
            self._using_backup = False

        self._update_visualization(phi)
        return np.array([[u_safe]])
