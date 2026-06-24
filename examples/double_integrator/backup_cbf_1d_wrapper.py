"""
Backup CBF wrapper for the 1D double integrator.

Subclasses BackupCBF from position_control/backup_cbf_qp.py and specializes it for:
  - State x = [s, v], safety h(x) = s >= 0
  - Input u in [u_min, u_max]  (stored in robot_spec['u_max'])
  - Backup control u_b in (0, u_max]  (constant positive acceleration)
  - Terminal set S_b = {s > 0, v > 0}, h_S(x) = min(s - eps_s, v - eps_v)
  - Callable alpha functions (not just linear coefficients)
  - Exact analytical flow map and sensitivity matrix (linear system, A^2 = 0)

The analytical expressions (generalized for any u_b):
  phi(t, x0) = [s0 + v0*t + u_b*t^2/2,  v0 + u_b*t]
  S(t)       = e^{A*t}  = [[1, t], [0, 1]]   (exact since A^2 = 0, independent of u_b)

Invariant sets (see run_backup_cbf_1d.py):
  S_max = { s > 0 if v >= 0,  s >= v^2/(2*u_max) if v < 0 }
  S(T)  = S_max conditions AND v >= -u_b*T
"""

import numpy as np
import cvxpy as cp
from typing import Callable, Optional
from safe_control.position_control.backup_cbf_qp import BackupCBF


class BackupCBF1D(BackupCBF):
    """
    Backup CBF for the 1D double integrator.

    Key overrides vs parent:
      - _alpha, _alpha_terminal: dispatch to user-supplied callable
      - _backup_control: constant u_b, no backup_controller object needed
      - _h_safety, _grad_h_safety: h(x) = s (analytical)
      - _h_terminal, _grad_h_terminal: min(s-eps_s, v-eps_v) (analytical)
      - _integrate_backup_trajectory: exact formula for linear system
      - solve_control_problem: 1D QP (scalar control variable)
    """

    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.02,
        backup_horizon: float = 2.0,
        u_b: float = 1.0,
        alpha_fn: Optional[Callable] = None,
        alpha_terminal_fn: Optional[Callable] = None,
        eps_s: float = 0.05,
        eps_v: float = 0.05,
        ax=None,
    ):
        """
        Args:
            robot: DoubleIntegrator1D instance
            robot_spec: robot specification dict (must contain 'u_max')
            dt: simulation timestep
            backup_horizon: backup horizon T (seconds)
            u_b: constant backup control value
            alpha_fn: class-K function for safety CBF, e.g. lambda h: h + h**3
            alpha_terminal_fn: class-K function for terminal CBF
            eps_s: position margin for terminal set h_S
            eps_v: velocity margin for terminal set h_S
            ax: matplotlib axis for visualization (optional)
        """
        super().__init__(robot, robot_spec, dt, backup_horizon, ax)

        # Correct dimensions (parent guesses 2D from unknown model key)
        self.n_states = 2
        self.n_controls = 1
        self.Q_u = np.array([1.0])

        self.u_b = u_b
        self.eps_s = eps_s
        self.eps_v = eps_v

        self._alpha_fn = alpha_fn if alpha_fn is not None else (lambda h: h)
        self._alpha_terminal_fn = (
            alpha_terminal_fn if alpha_terminal_fn is not None else (lambda h: h)
        )

    # ------------------------------------------------------------------
    # Alpha overrides: callable instead of scalar coefficient
    # ------------------------------------------------------------------

    def _alpha(self, h: float) -> float:
        return float(self._alpha_fn(h))

    def _alpha_terminal(self, h: float) -> float:
        return float(self._alpha_terminal_fn(h))

    # ------------------------------------------------------------------
    # Backup control: constant u_b, no backup_controller object needed
    # ------------------------------------------------------------------

    def _backup_control(self, x) -> np.ndarray:
        return np.array([self.u_b])

    # ------------------------------------------------------------------
    # Safety CBF: h(x) = s  (analytical gradient)
    # ------------------------------------------------------------------

    def _h_safety(self, x, t: float = 0.0) -> float:
        return float(np.array(x).flatten()[0])

    def _grad_h_safety(self, x, t: float = 0.0) -> np.ndarray:
        return np.array([1.0, 0.0])

    # ------------------------------------------------------------------
    # Terminal CBF: h_S(x) = min(s - eps_s, v - eps_v)
    # ------------------------------------------------------------------

    def _h_terminal(self, x) -> float:
        xf = np.array(x).flatten()
        return float(min(xf[0] - self.eps_s, xf[1] - self.eps_v))

    def _grad_h_terminal(self, x) -> np.ndarray:
        xf = np.array(x).flatten()
        if (xf[0] - self.eps_s) <= (xf[1] - self.eps_v):
            return np.array([1.0, 0.0])   # s - eps_s is active
        else:
            return np.array([0.0, 1.0])   # v - eps_v is active

    # ------------------------------------------------------------------
    # Trajectory integration: analytical (exact for linear system)
    # ------------------------------------------------------------------

    def _integrate_backup_trajectory(self, x0) -> tuple:
        """Override: use analytical formula instead of numerical Euler."""
        return self._integrate_backup_trajectory_analytical(x0)

    def _integrate_backup_trajectory_analytical(self, x0) -> tuple:
        """
        Exact flow map and sensitivity for the linear 1D double integrator.

        phi(t) = [s0 + v0*t + u_b*t^2/2,  v0 + u_b*t]
        S(t)   = e^{A*t} = [[1, t], [0, 1]]   (exact since A^2 = 0)

        Returns:
            phi: (N, 2) state trajectory under backup control
            S:   (N, 2, 2) sensitivity matrices (STM)
        """
        x0f = np.array(x0).flatten()
        s0, v0 = x0f[0], x0f[1]

        t = np.arange(self.N) * self.dt   # (N,)

        phi = np.zeros((self.N, 2))
        phi[:, 0] = s0 + v0 * t + 0.5 * self.u_b * t**2
        phi[:, 1] = v0 + self.u_b * t

        S = np.zeros((self.N, 2, 2))
        S[:, 0, 0] = 1.0
        S[:, 0, 1] = t
        S[:, 1, 0] = 0.0
        S[:, 1, 1] = 1.0

        return phi, S

    def integrate_backup_trajectory_numerical(self, x0) -> tuple:
        """Numerical backup trajectory via parent's Euler + finite-diff sensitivity."""
        return super()._integrate_backup_trajectory(x0)

    def compare_trajectories(self, x0) -> dict:
        """
        Compare analytical and numerical backup trajectory from x0.

        For this nilpotent linear system, the Euler sensitivity IS exact
        (S_numerical == S_analytical), but the state has O(dt) per-step error.

        Returns dict with keys:
            phi_analytical, S_analytical, phi_numerical, S_numerical,
            phi_error_norm (N,), S_error_norm (N,), max_phi_error, max_S_error
        """
        phi_a, S_a = self._integrate_backup_trajectory_analytical(x0)
        phi_n, S_n = self.integrate_backup_trajectory_numerical(x0)

        phi_err = np.linalg.norm(phi_a - phi_n, axis=1)
        S_err = np.array([
            np.linalg.norm(S_a[i] - S_n[i], 'fro') for i in range(self.N)
        ])

        return {
            'phi_analytical': phi_a,
            'S_analytical': S_a,
            'phi_numerical': phi_n,
            'S_numerical': S_n,
            'phi_error_norm': phi_err,
            'S_error_norm': S_err,
            'max_phi_error': float(np.max(phi_err)),
            'max_S_error': float(np.max(S_err)),
        }

    # ------------------------------------------------------------------
    # QP solver: 1D specialization (scalar control variable)
    # ------------------------------------------------------------------

    def solve_control_problem(self, robot_state, friction=None):
        """
        Backup CBF-QP for 1D double integrator (scalar control u ∈ [-1, 1]).

        Builds N-1 safety constraints + 1 terminal constraint and solves:
            minimize   (u - u_nom)^2
            subject to  t_i * u >= rhs_i   for i = 1..N-1
                        terminal constraint
                        -u_max <= u <= u_max
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

        # Dynamics at current state
        f0 = self._dynamics_f(robot_state)   # (2,)
        g0 = self._dynamics_g(robot_state)   # (2, 1)

        # Build constraint lists: coeff_i * u >= rhs_i
        coeffs = []
        rhs_vals = []

        for i in range(1, self.N):
            x_i = phi[i]
            S_i = S[i]
            t_i = i * self.dt

            h_val = self._h_safety(x_i, t_i)
            grad_h = self._grad_h_safety(x_i, t_i)   # (2,)

            # Backup drift: d/dt phi at time t_i
            if i < self.N - 1:
                f_pi = (phi[i + 1] - phi[i]) / self.dt
            else:
                f_pi = (phi[i] - phi[i - 1]) / self.dt

            # lhs coefficient: grad_h @ S_i @ g0  (scalar for 1D)
            c = float((grad_h @ S_i @ g0).flat[0])
            # rhs: -grad_h @ S_i @ f0 + grad_h @ f_pi - alpha(h)
            r = float(-(grad_h @ S_i @ f0) + (grad_h @ f_pi) - self._alpha(h_val))

            if abs(c) > 1e-6:
                coeffs.append(c)
                rhs_vals.append(r)

        # Terminal constraint
        x_T, S_T = phi[-1], S[-1]
        h_terminal = self._h_terminal(x_T)
        grad_hT = self._grad_h_terminal(x_T)
        c_T = float((grad_hT @ S_T @ g0).flat[0])
        r_T = float(-(grad_hT @ S_T @ f0 + self._alpha_terminal(h_terminal)))

        if abs(c_T) > 1e-6:
            coeffs.append(c_T)
            rhs_vals.append(r_T)

        # Solve 1D QP
        if len(coeffs) > 0:
            coeffs_arr = np.array(coeffs)    # (n,)
            rhs_arr = np.array(rhs_vals)     # (n,)

            u_s = cp.Variable()              # scalar variable
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
                    u_safe = float(self.u_b)
                    self._last_intervention = True
                    self._using_backup = True
        else:
            u_safe = u_nom
            self._last_intervention = False
            self._using_backup = False

        self._update_visualization(phi)
        return np.array([[u_safe]])
