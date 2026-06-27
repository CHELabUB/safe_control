"""
Sandwiched backup CBF for the ego between a (mildly braking) lead and an interactive rear.

The backup-CBF counterpart to the analytic SandwichedHOCBF (`sandwiched_cbf.py`). The no-lead
RearEndBackupCBF1D escapes by *accelerating to v_max*; that is impossible here -- the lead blocks
the ego -- so the backup is instead a **rear-aware car-following OVM**: the ego follows the lead
but *brakes less* when the rear is catching up, trading front gap for rear safety.

    a_backup = alpha_b*(V_h(h_f) - v_e) + beta_b*(W(v_l) - v_e)   # follow the lead (OVM)
             + beta_f*(v_rear - v_e)                              # rear-aware softening
    a_backup = clip(a_backup, -a_e, u_acc)

When v_rear > v_e (rear approaching) the beta_f term is positive and reduces the brake. This is
the mechanism that makes interaction-model *accuracy decide safety*:

  accurate (sluggish) rear -> in the rollout v_rear stays high -> beta_f*(v_r - v_e) is large ->
        the backup brakes very gently -> the backup trajectory eats forward gap -> the (speed-
        dependent) forward barrier binds earlier -> the ego hangs back PROACTIVELY and slows early
        -> it has room, so the eventual brake is gentle enough for the real rear -> safe both.
  over-estimated rear      -> the assumed rear brakes hard in the rollout, v_rear drops, the
        beta_f softening vanishes -> the backup brakes normally -> small forward gap -> the ego
        tailgates -> when the lead slows it must brake hard -> the real sluggish rear is rear-ended.

Plant (reused from rear_backup_cbf): the 4-state augmented system

    x = [s_ego, v_ego, s_rear, v_rear]      f = [v_e, 0, v_r, a_rear(x)]^T,  g = [0,1,0,0]^T

with the rear's reaction folded into the drift, so the base BackupCBF._integrate_backup_trajectory
gives a rigorous flow + sensitivity (the STM captures the rear reacting to an ego perturbation).
The **lead is exogenous** (it does not react), supplied via set_lead_state() and predicted
worst-case (brake-to-stop at -a_l) over the rollout -- mirroring CarFollowingBackupCBF1D. Two
barriers are imposed along the rollout:

    rear (in-state):  h_r(x)   = (s_e - L) - s_r - d_min,                grad [1, 0, -1, 0]
    forward (lead):   h_f(x,t) = (s_l(t) - L) - s_e - b_hat(v_e, v_l(t)) - d_f,
                      grad [-1, -db_hat/dv_e, 0, 0]   (+ dh_f/dt for the moving lead)

`b_hat` is the He-Orosz maximal-invariant safe distance (`car_following_safe_distance`), the same
surface the Stage-A forward CBF uses -- so the forward barrier keeps a real *speed-dependent*
following gap (no tailgating to a fixed margin). The QP is **forward-priority**: the forward
safety slack carries a much larger penalty than the rear's, so the ego never hits the lead; under
a genuine conflict it brakes for the lead and the rear may be hit (the same resolution as Stage A).

Because the backup policy depends on the lead at rollout time t, the time-agnostic base
`_backup_control(x)` is replaced by a time-threaded rollout (`_integrate_backup_trajectory`
override) that passes t = i*dt into `_backup_control_t(x, t)`; the finite-difference perturbations
reuse the same t. Terminal constraints are OFF by default (`use_terminal=False`): the per-step
forward+rear safety constraints along a long horizon carry the demonstration.
"""

import os
import sys

import numpy as np
import cvxpy as cp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'car_following')))

from safe_control.position_control.backup_cbf_qp import BackupCBF       # noqa: E402
from car_following_models import ovm_accel                              # noqa: E402
from car_following_safe_distance import safe_distance, safe_distance_grad  # noqa: E402
from rear_backup_cbf import _RearAwareAugmentedDI, BODY_LENGTH          # noqa: E402


class SandwichedBackupCBF1D(BackupCBF):
    """Backup CBF with a rear-aware OVM backup, between a lead and an interactive follower."""

    def __init__(self, robot_spec, p_safe, dt=0.05, backup_horizon=6.0,
                 a_e=3.0, u_acc=3.0, L=4.5, d_min=1.0, d_f=0.5,
                 beta_f=0.5, backup_ovm_params=None,
                 rear_model='ovm', rear_params=None,
                 gamma=1.0, gamma_terminal=2.0, use_terminal=False, ax=None):
        """
        Args:
            robot_spec: dict (needs 'u_max' for QP scaling; set to u_acc).
            p_safe: He-Orosz safe-distance params {tau, a_e, a_l, vbar} for the forward barrier.
            backup_horizon: rollout horizon T (long enough for the OVM to settle).
            a_e: ego max deceleration (QP brake bound). u_acc: ego max acceleration.
            L: combined center-to-bumper offset. d_min/d_f: rear/extra-forward margins.
            beta_f: rear-aware softening gain in the backup OVM (the key knob).
            backup_ovm_params: OVM params {alpha, beta, kappa, h_st, v_max} for the
                follow-the-lead part of the backup (moderate following law by default).
            rear_model, rear_params: the ego's ASSUMED model of the rear (may be wrong);
                folded into the augmented dynamics.
            gamma, gamma_terminal: class-K gains. use_terminal: include the gap terminals.
        """
        self.p_safe = dict(p_safe)
        self.a_l = float(self.p_safe['a_l'])
        self.a_e = float(abs(a_e))
        self.u_acc = float(abs(u_acc))
        self.L = float(L)
        self.d_min = float(d_min)
        self.d_f = float(d_f)
        self.beta_f = float(beta_f)
        self.backup_ovm = dict(backup_ovm_params or
                               {'alpha': 0.6, 'beta': 0.5, 'kappa': 0.5,
                                'h_st': 5.0, 'v_max': 13.0})

        aug_spec = dict(robot_spec)
        aug_spec.setdefault('model', 'DoubleIntegrator1D')
        aug_spec.setdefault('u_max', self.u_acc)
        aug_robot = _RearAwareAugmentedDI(dt, rear_model, dict(rear_params or {}),
                                          v_max=aug_spec.get('v_max', 13.0),
                                          body_length=BODY_LENGTH, robot_spec=aug_spec)
        super().__init__(aug_robot, aug_spec, dt=dt, backup_horizon=backup_horizon, ax=ax)

        self.n_states = 4
        self.n_controls = 1
        self.Q_u = np.array([1.0])
        self.alpha = float(gamma)
        self.alpha_terminal = float(gamma_terminal)
        self.use_terminal = bool(use_terminal)
        self.u_b = -self.a_e                        # QP fallback = hard brake (forward-priority)
        self._rear0 = (0.0, 0.0)
        self._lead0 = (1e6, 0.0)                    # (s_lead, v_lead) captured each step
        self._rear_s = self._rear_v = None
        self.latest_backup_trajectory = None
        self.last_status = 'none'
        # Per-step soft-constraint violation (max slack) by priority group; nan = QP fallback.
        self.last_fwd_slack = 0.0       # forward barrier (priority, RHO_FWD)
        self.last_rear_slack = 0.0      # rear barrier (secondary, RHO_REAR)

    # ------------------------------------------------------------------
    # Per-step external state
    # ------------------------------------------------------------------

    def set_rear_state(self, s_rear, v_rear):
        self._rear0 = (float(s_rear), float(v_rear))

    def set_lead_state(self, s_lead, v_lead):
        """Capture the lead's current state; the rollout predicts it worst-case from here."""
        self._lead0 = (float(s_lead), float(v_lead))

    def _lead(self, t):
        """Predicted lead (s_l, v_l) at rollout time t: brake at -a_l to a stop, then hold."""
        s0, v0 = self._lead0
        t = max(float(t), 0.0)
        t_stop = v0 / self.a_l if self.a_l > 1e-9 else 0.0
        if t <= t_stop:
            return s0 + v0 * t - 0.5 * self.a_l * t * t, v0 - self.a_l * t
        return s0 + 0.5 * v0 * t_stop, 0.0          # stopped at s0 + v0^2/(2 a_l)

    # ------------------------------------------------------------------
    # Rear-aware OVM backup policy (time-aware)
    # ------------------------------------------------------------------

    def _backup_control_t(self, x, t):
        xf = np.array(x).flatten()
        s_e, v_e, _, v_r = (float(v) for v in xf)
        s_l, v_l = self._lead(t)
        ego_d = {'x': s_e, 'vx': v_e, 'length': BODY_LENGTH}
        lead_d = {'x': s_l, 'vx': v_l, 'length': BODY_LENGTH}
        a = ovm_accel(ego_d, lead_d, self.backup_ovm)       # follow the lead
        a += self.beta_f * (v_r - v_e)                       # rear-aware softening
        return np.array([float(np.clip(a, -self.a_e, self.u_acc))])

    def _backup_control(self, x):                            # base API (t=0 fallback)
        return self._backup_control_t(x, 0.0)

    # ------------------------------------------------------------------
    # Barriers
    # ------------------------------------------------------------------

    def _h_rear(self, x):
        xf = np.array(x).flatten()
        return (xf[0] - self.L) - xf[2] - self.d_min

    _GRAD_REAR = np.array([1.0, 0.0, -1.0, 0.0])
    _GRAD_FWD_S = np.array([-1.0, 0.0, 0.0, 0.0])           # position part of grad h_f

    def _h_fwd(self, x, t):
        xf = np.array(x).flatten()
        s_l, v_l = self._lead(t)
        return (s_l - self.L) - xf[0] - safe_distance(xf[1], v_l, self.p_safe) - self.d_f

    def _grad_h_fwd(self, x, t):
        xf = np.array(x).flatten()
        _, v_l = self._lead(t)
        db_dv, _ = safe_distance_grad(xf[1], v_l, self.p_safe)
        return np.array([-1.0, -db_dv, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Time-threaded backup rollout (flow + sensitivity); mirrors the base loop but
    # passes the rollout time t into the lead-dependent backup control.
    # ------------------------------------------------------------------

    def _integrate_backup_trajectory(self, x0):
        phi = np.zeros((self.N, self.n_states))
        S = np.zeros((self.N, self.n_states, self.n_states))
        x = np.array(x0).flatten().copy()
        S_curr = np.eye(self.n_states)
        phi[0] = x
        S[0] = S_curr
        eps = 1e-5
        n = self.n_states
        for i in range(1, self.N):
            t = (i - 1) * self.dt                        # time at the *start* of this step
            u_b = self._backup_control_t(x, t)
            x_next = np.array(self.robot.step(x.reshape(-1, 1), u_b.reshape(-1, 1))).flatten()
            A = np.zeros((n, n))
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] += eps
                u_pert = self._backup_control_t(x_pert, t)
                x_np = np.array(self.robot.step(x_pert.reshape(-1, 1),
                                                u_pert.reshape(-1, 1))).flatten()
                A[:, j] = (x_np - x_next) / eps
            S_curr = A @ S_curr
            x = x_next
            phi[i] = x
            S[i] = S_curr
        return phi, S

    # ------------------------------------------------------------------
    # QP: forward (priority) + rear safety along the rollout (soft), optional terminals.
    # ------------------------------------------------------------------

    def solve_control_problem(self, robot_state, friction=None):
        xf = np.array(robot_state).flatten()
        x0 = np.array([xf[0], xf[1], self._rear0[0], self._rear0[1]])

        phi, S = self._integrate_backup_trajectory(x0)
        self._rear_s, self._rear_v = phi[:, 2], phi[:, 3]
        self.latest_backup_trajectory = phi[:, :2].copy()

        h_r_vals = [self._h_rear(phi[i]) for i in range(self.N)]
        h_f_vals = [self._h_fwd(phi[i], i * self.dt) for i in range(self.N)]
        self._last_h_min = min(min(h_r_vals), min(h_f_vals))
        if self._last_h_min < self.global_min_h:
            self.global_min_h = self._last_h_min
        self.curr_step += 1

        u_max = self.robot_spec.get('u_max', 1.0)
        u_nom = float(np.clip(self._get_nominal_control(x0), -u_max, u_max).flat[0])

        f0 = self._dynamics_f(x0)
        g0 = self._dynamics_g(x0)
        grad_r = self._GRAD_REAR

        fwd, rear = [], []                              # (coeff, rhs) per constraint group
        for i in range(1, self.N):
            x_i, S_i, t_i = phi[i], S[i], i * self.dt
            if i < self.N - 1:
                f_pi = (phi[i + 1] - phi[i]) / self.dt
            else:
                f_pi = (phi[i] - phi[i - 1]) / self.dt
            # Rear barrier (in-state -> no exogenous dh/dt).
            c = float(grad_r @ S_i @ g0)
            if abs(c) > 1e-6:
                r = float(-(grad_r @ S_i @ f0) + (grad_r @ f_pi) - self._alpha(self._h_rear(x_i)))
                rear.append((c, r))
            # Forward barrier (speed-dependent, moving lead -> dh_f/dt term).
            grad_f = self._grad_h_fwd(x_i, t_i)
            cf = float(grad_f @ S_i @ g0)
            if abs(cf) > 1e-6:
                hf = self._h_fwd(x_i, t_i)
                dhf_dt = (self._h_fwd(x_i, t_i + self.dt) - hf) / self.dt
                rf = float(-(grad_f @ S_i @ f0) + (grad_f @ f_pi) - dhf_dt - self._alpha(hf))
                fwd.append((cf, rf))

        # Optional terminal gap constraints (off by default).
        term_fwd, term_rear = [], []
        if self.use_terminal:
            x_T, S_T, t_T = phi[-1], S[-1], (self.N - 1) * self.dt
            cR = float(grad_r @ S_T @ g0)
            if abs(cR) > 1e-6:
                term_rear.append((cR, float(-(grad_r @ S_T @ f0)
                                            - self._alpha_terminal(self._h_rear(x_T)))))
            grad_fT = self._grad_h_fwd(x_T, t_T)
            cF = float(grad_fT @ S_T @ g0)
            if abs(cF) > 1e-6:
                hT = self._h_fwd(x_T, t_T)
                dhT = (self._h_fwd(x_T, t_T + self.dt) - hT) / self.dt
                term_fwd.append((cF, float(-(grad_fT @ S_T @ f0) - dhT
                                           - self._alpha_terminal(hT))))

        # Forward-priority: forward safety slack penalized far more than the rear's.
        RHO_FWD, RHO_REAR, RHO_TERM = 1e5, 1e3, 1e1
        status = 'no_constraints'
        # (name, constraints, penalty, priority-bucket): priority = forward, secondary = rear.
        group_specs = [('fwd', fwd, RHO_FWD, 'fwd'), ('rear', rear, RHO_REAR, 'rear'),
                       ('term_fwd', term_fwd, RHO_TERM, 'fwd'),
                       ('term_rear', term_rear, RHO_TERM, 'rear')]
        self.last_fwd_slack = self.last_rear_slack = 0.0
        if any(g for _, g, _, _ in group_specs):
            u_s = cp.Variable()
            objective = (u_s - u_nom / u_max) ** 2
            constraints = [u_s >= -1.0, u_s <= 1.0]
            slack_of = {}                           # bucket -> list of slack cp.Variables
            for name, grp, rho, bucket in group_specs:
                if not grp:
                    continue
                ca = np.array([c for c, _ in grp])
                ra = np.array([r for _, r in grp])
                xi = cp.Variable(len(grp), nonneg=True)
                constraints.append((ca * u_max) * u_s + xi >= ra)
                objective = objective + rho * cp.sum_squares(xi)
                slack_of.setdefault(bucket, []).append(xi)
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
                self._last_intervention = abs(u_safe - u_nom) > 0.01 * u_max
                self._using_backup = self._last_intervention
                # Worst soft-constraint violation per priority bucket (max slack >= 0).
                def _max_slack(bucket):
                    xis = slack_of.get(bucket, [])
                    vals = [float(np.max(x.value)) for x in xis if x.value is not None]
                    return max(vals) if vals else 0.0
                self.last_fwd_slack = _max_slack('fwd')
                self.last_rear_slack = _max_slack('rear')
            else:
                u_safe = float(self.u_b)            # forward-priority hard brake
                self._last_intervention = True
                self._using_backup = True
                self.last_fwd_slack = self.last_rear_slack = float('nan')   # QP fallback
        else:
            u_safe = u_nom
            self._last_intervention = False
            self._using_backup = False

        self.last_status = status
        return np.array([[u_safe]])
