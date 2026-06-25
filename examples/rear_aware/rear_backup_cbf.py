"""
Rear-end-avoidance Backup CBF for the ego, on the *augmented* rear-aware system.

Mirror image of the forward CarFollowingBackupCBF1D: instead of braking to a stop
behind a lead, the ego *accelerates to v_max* to escape an unfiltered follower behind
it. Unlike the forward case (where the lead is an exogenous signal that does not enter
the dynamics), the rear vehicle **reacts to the ego**, so it must be part of the state.

This module therefore augments the plant to the joint state

    x = [s_ego, v_ego, s_rear, v_rear]

and folds the rear's car-following reaction a_rear(x) into the drift, giving a
control-affine system  x_dot = f(x) + g(x) u  with the environment policy *inside* f:

    f(x) = [v_ego, 0, v_rear, a_rear(x)]^T      g(x) = [0, 1, 0, 0]^T

The backup flow and its sensitivity matrix S_i = d phi_i / d x0 are then produced by the
**base BackupCBF._integrate_backup_trajectory** (rollout via robot.step + finite-difference
of the one-step dynamics) -- a rigorous flow sensitivity, not an end-to-end barrier
finite difference. Because the rear is in the state, the STM captures the rear reacting to
an ego perturbation automatically.

  - Backup policy: ego accelerates to v_max then holds (`_ego_backup_accel`, the hook for
    an environment-aware escape that may depend on the rear).
  - Barrier:  h(x) = (s_ego - L) - s_rear - d_min,  grad h = [1, 0, -1, 0]  (exact, constant).
  - Terminal set: positive gap at the horizon (same h at phi[-1]); optional (`use_terminal`).

QP (mirrors CarFollowingBackupCBF1D): for each backup step i,
    (grad_h . S_i . g0) u  >=  -(grad_h . S_i . f0) + grad_h . f_pi_i - alpha(h_i)
with no explicit dh/dt term -- the rear is a state now, not an exogenous obstacle.
"""

import os
import sys

import numpy as np
import cvxpy as cp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'car_following')))

from safe_control.position_control.backup_cbf_qp import BackupCBF   # noqa: E402
from car_following_models import nominal_accel                      # noqa: E402

BODY_LENGTH = 4.5


class _RearAwareAugmentedDI:
    """Augmented 4-state plant [s_e, v_e, s_r, v_r] for the rear-aware backup CBF.

    The ego is a double integrator driven by the control u (its acceleration); the rear
    vehicle's car-following reaction a_rear(x) is folded into the drift, so the closed-loop
    plant is control-affine  x_dot = f(x) + g(x) u  with the environment policy inside f.
    The base BackupCBF then rolls this out and finite-differences it for the (rigorous)
    flow sensitivity matrix.
    """

    def __init__(self, dt, rear_model, rear_params, v_max,
                 body_length=BODY_LENGTH, robot_spec=None):
        self.dt = float(dt)
        self.rear_model = rear_model
        self.rear_params = dict(rear_params or {})
        self.v_max = float(v_max)
        self.body_length = float(body_length)
        self.robot_spec = robot_spec or {}

    def _a_rear(self, X):
        s_e, v_e, s_r, v_r = (float(v) for v in np.array(X).flatten())
        ego_d = {'x': s_e, 'vx': v_e, 'length': self.body_length}
        rear_d = {'x': s_r, 'vx': v_r, 'length': self.body_length}
        return nominal_accel(self.rear_model, rear_d, ego_d, self.rear_params)

    def f(self, X):
        """Drift f(x) = [v_e, 0, v_r, a_rear(x)]^T  (rear policy folded in)."""
        xf = np.array(X).flatten()
        return np.array([[xf[1]], [0.0], [xf[3]], [self._a_rear(xf)]])

    def g(self, X):
        """Control matrix g(x) = [0, 1, 0, 0]^T (ego acceleration)."""
        return np.array([[0.0], [1.0], [0.0], [0.0]])

    def step(self, X, U):
        """Euler step with the sim's clamps (ego v in [0, v_max], rear v >= 0)."""
        X = np.array(X).reshape(-1, 1)
        U = np.array(U).reshape(-1, 1)
        Xn = (X + (self.f(X) + self.g(X) @ U) * self.dt).flatten()
        Xn[1] = min(max(Xn[1], 0.0), self.v_max)   # ego: no reverse, capped at v_max
        Xn[3] = max(Xn[3], 0.0)                     # rear: no reverse
        return Xn.reshape(-1, 1)


class RearEndBackupCBF1D(BackupCBF):
    """Backup CBF that accelerates the ego to v_max to avoid being rear-ended.

    Subclasses the base BackupCBF so the backup flow + sensitivity matrix come from the
    framework's own `_integrate_backup_trajectory` (no bespoke rollout / no end-to-end
    barrier finite difference). Only the augmented barrier, the escape backup control, and
    a scalar soft QP are specialized here.
    """

    def __init__(self, robot, robot_spec, dt=0.05, backup_horizon=5.0,
                 u_acc=3.0, v_max=12.0, L=4.5, d_min=1.0,
                 rear_model='ovm', rear_params=None,
                 gamma=1.0, gamma_terminal=2.0, use_terminal=True, ax=None):
        """
        Args:
            robot: unused (kept for call-site compatibility); the augmented plant is
                built internally from the params below.
            robot_spec: must contain 'u_max' (= u_acc for QP scaling/bounds).
            backup_horizon: horizon T; should exceed (v_max - 0)/u_acc so the ego
                actually reaches v_max within the rollout.
            u_acc: backup acceleration magnitude (escape).
            v_max: speed the ego accelerates to.
            L, d_min: combined length and rear collision margin.
            rear_model, rear_params: the ego's *assumed* model of the rear (may differ
                from the rear's true params); folded into the augmented dynamics.
            gamma, gamma_terminal: class-K gains for the safety / terminal constraints.
            use_terminal: include the terminal (gap-at-horizon) constraint. When False the
                QP keeps only the per-step horizon safety constraints.
        """
        self.u_acc = float(abs(u_acc))
        self.v_max = float(v_max)
        self.L = float(L)
        self.d_min = float(d_min)
        self.rear_model = rear_model
        self.rear_params = dict(rear_params or {})

        # Augmented plant with the rear policy folded into the dynamics. Use the generic
        # f/g path in the base by keeping a non-special model key.
        aug_spec = dict(robot_spec)
        aug_spec.setdefault('model', 'DoubleIntegrator1D')
        aug_robot = _RearAwareAugmentedDI(dt, self.rear_model, self.rear_params,
                                          self.v_max, BODY_LENGTH, aug_spec)
        super().__init__(aug_robot, aug_spec, dt=dt, backup_horizon=backup_horizon, ax=ax)

        self.n_states = 4
        self.n_controls = 1
        self.Q_u = np.array([1.0])
        self.alpha = float(gamma)                  # base _alpha returns alpha * h
        self.alpha_terminal = float(gamma_terminal)
        self.use_terminal = bool(use_terminal)
        self.u_b = +self.u_acc                      # escape accel (QP fallback)
        self._rear0 = (0.0, 0.0)                    # current rear state (s_r, v_r)
        self._rear_s = None
        self._rear_v = None
        self.last_status = 'none'

    def set_rear_state(self, s_rear: float, v_rear: float):
        """Provide the rear vehicle's current state before solve_control_problem."""
        self._rear0 = (float(s_rear), float(v_rear))

    # ------------------------------------------------------------------
    # Backup (escape) control and augmented barrier
    # ------------------------------------------------------------------

    def _ego_backup_accel(self, ego_d, rear_d):
        """Ego backup acceleration, evaluated on the joint state (ego_d, rear_d).
        CRH: Default escape policy: accelerate toward v_max then hold. 
        """
        return self.u_acc if ego_d['vx'] < self.v_max else 0.0

    def _h_terminal(self, x) -> float:
        """
        Terminal gap at the horizon (no rear-end collision).
        CRH: This is associated with maximum acceleration to v_max. 
        Assuming the rear is having the same limit, and initially _h_safety(x) >= 0, 
        then _h_safety(x) should be non-decreasing.
        This means that we could add this terminal constraints and it still works.
        """
        return self._h_safety(x)

    def _grad_h_terminal(self, x) -> np.ndarray:
        return np.array([1.0, 0.0, -1.0, 0.0])

    def _backup_control(self, x):
        xf = np.array(x).flatten()
        ego_d = {'x': xf[0], 'vx': xf[1], 'length': BODY_LENGTH}
        rear_d = {'x': xf[2], 'vx': xf[3], 'length': BODY_LENGTH}
        return np.array([self._ego_backup_accel(ego_d, rear_d)])

    # ------------------------------------------------------------------
    # safety requirement: 
    # h(x) = (s_ego - L) - s_rear - d_min >= 0 no rear-end collision
    # ------------------------------------------------------------------

    def _h_safety(self, x, t: float = 0.0) -> float:
        xf = np.array(x).flatten()
        return (xf[0] - self.L) - xf[2] - self.d_min

    def _grad_h_safety(self, x, t: float = 0.0) -> np.ndarray:
        return np.array([1.0, 0.0, -1.0, 0.0])

    # ------------------------------------------------------------------
    # QP (soft-constrained scalar; mirrors CarFollowingBackupCBF1D) using the
    # base-class backup flow + sensitivity matrix.
    # ------------------------------------------------------------------

    def solve_control_problem(self, robot_state, friction=None):
        xf = np.array(robot_state).flatten()
        # Assemble the augmented initial state from the ego state + stored rear state.
        x0 = np.array([xf[0], xf[1], self._rear0[0], self._rear0[1]])

        # Rigorous flow + sensitivity from the base class (rollout via robot.step,
        # STM via finite-difference of the one-step augmented dynamics).
        phi, S = self._integrate_backup_trajectory(x0)
        self._rear_s, self._rear_v = phi[:, 2], phi[:, 3]

        h_vals = [self._h_safety(phi[i]) for i in range(self.N)]
        self._last_h_min = min(float(np.min(h_vals)), self._h_terminal(phi[-1]))
        if self._last_h_min < self.global_min_h:
            self.global_min_h = self._last_h_min
        self.latest_backup_trajectory = phi[:, :2].copy()
        self.curr_step += 1

        u_nom = self._get_nominal_control(x0)
        u_max = self.robot_spec.get('u_max', 1.0)
        u_nom = float(np.clip(u_nom, -u_max, u_max).flat[0])

        f0 = self._dynamics_f(x0)              # (4,)  drift incl. rear reaction
        g0 = self._dynamics_g(x0)              # (4, 1)
        grad_h = self._grad_h_safety(phi[0])   # constant [1, 0, -1, 0]

        coeffs, rhs_vals = [], []
        for i in range(1, self.N):
            x_i, S_i = phi[i], S[i]
            h_val = self._h_safety(x_i)
            if i < self.N - 1:
                f_pi = (phi[i + 1] - phi[i]) / self.dt
            else:
                f_pi = (phi[i] - phi[i - 1]) / self.dt
            c = float(grad_h @ S_i @ g0)
            r = float(-(grad_h @ S_i @ f0) + (grad_h @ f_pi) - self._alpha(h_val))
            if abs(c) > 1e-6:
                coeffs.append(c)
                rhs_vals.append(r)

        # Terminal: gap at the horizon (no f_pi drift term, no exogenous dh/dt).
        x_T, S_T = phi[-1], S[-1]
        hT = self._h_terminal(x_T)
        grad_hT = self._grad_h_terminal(x_T)
        c_T = float(grad_hT @ S_T @ g0)
        r_T = float(-(grad_hT @ S_T @ f0) - self._alpha_terminal(hT))
        has_terminal = self.use_terminal and abs(c_T) > 1e-6

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
