"""
Sandwiched front+rear safety filter for the ego between a lead and a follower.

The ego must stay safe behind a **lead** (forward CBF, an upper bound on its acceleration)
*and* avoid being **rear-ended** by an interactive follower (rear HOCBF, a lower bound). Unlike
the no-lead ego_rear scenario, the ego cannot escape forward -- the lead blocks it -- so the
accuracy of the assumed rear model now decides *safety*, not just performance.

**Rear-aware forward CBF (a_e_eff coupling).** A naive front-ceiling / rear-floor clip is *not*
enough: an aggressive nominal tailgates to the forward boundary, the forward CBF then demands a
hard brake regardless of the rear, and the rear floor only limits braking *reactively* -- so the
ego rear-ends the follower whether or not the rear model is accurate. The fix is to let the
**assumed rear model set the ego's effective braking authority** for *forward* planning:

    a_e_eff = largest ego deceleration for which the ASSUMED rear keeps h_r >= d_min
              (a sudden-stop probe: ego brakes at a const a_e to rest while the assumed rear
               follows; bisect a_e until the worst-case rear gap just touches d_min).

`a_e_eff` is fed **only into the forward CBF safe-distance** (b_hat grows as a_e shrinks), so it
controls *how far back the ego plans to sit*, not its actuator authority. Rear protection stays
with the rear HOCBF floor `lb_r`; the ego retains full physical braking for genuine forward
emergencies (forward-priority). The point is the *planned gap*, not the brake clamp:

  accurate (sluggish) rear -> small a_e_eff -> large forward safe-distance -> the ego hangs back
                              and slows early -> it has room, so the honest rear floor's gentle
                              brake keeps it forward-safe too -> safe on BOTH fronts.
  over-estimated rear      -> large a_e_eff -> small forward safe-distance -> the ego tailgates;
                              when the lead slows it must brake hard, the permissive rear floor
                              lets that through, and the real (sluggish) rear is rear-ended.

(Clamping the *actuator* to a_e_eff instead was tried and is wrong: with a sluggish, nearby rear
a_e_eff collapses, starving the ego of the braking it needs for the lead -> front collisions.)

Combined analytic bound (no QP), per step:

    a_e_eff = probe(assumed rear, h_r, v_ego, v_rear)   # sets the forward hang-back distance
    rhs_f   = forward CBF upper bound, b_hat built with a_e_eff   (CarFollowingCBF1D vs lead)
    lb_r    = rear HOCBF lower bound (CoupledRearCBF, ASSUMED rear)   # rear protection
    a = clip(a_nom, lb_r, rhs_f);  if lb_r > rhs_f: a = rhs_f  (forward-priority)
    a = clip(a, -a_e, a_acc)                            # full physical braking authority

Reuses CarFollowingCBF1D (`accel_upper_bound`) and CoupledRearCBF (`lower_bound`, `rear_accel`).
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'car_following')))

from car_following_cbf import CarFollowingCBF1D          # noqa: E402
from coupled_rear_cbf import CoupledRearCBF              # noqa: E402

BODY_LENGTH = 4.5


class SandwichedHOCBF:
    """Forward CBF (vs lead) + coupled rear HOCBF (vs follower), combined analytically."""

    def __init__(self, forward_spec, p_safe, rear_model, rear_params,
                 gamma=1.0, L=4.5, d_min=1.0, a_e=3.0, a_acc=3.0,
                 alpha1=1.0, alpha2=1.0, robust_factor=1.0,
                 couple_a_e_eff=True, a_e_eff_min=0.5,
                 probe_dt=0.05, probe_t_max=8.0, probe_iters=18):
        """
        Args:
            forward_spec, p_safe, gamma, L: forwarded to CarFollowingCBF1D (forward CBF vs
                the lead; `p_safe` holds tau/a_e/a_l/vbar).
            rear_model, rear_params: the ego's ASSUMED model of the rear (may be wrong);
                forwarded to CoupledRearCBF.
            d_min, a_e, a_acc, alpha1, alpha2, robust_factor: rear HOCBF params (a in
                [-a_e, a_acc]).
            couple_a_e_eff: if True (default) the forward CBF's braking authority (and the
                actual brake clamp) is set per-step to `a_e_eff`, the largest decel the
                ASSUMED rear survives (sudden-stop probe). If False, falls back to the naive
                fixed-a_e front-ceiling / rear-floor clip.
            a_e_eff_min: floor on a_e_eff (keep the forward CBF well-posed, a_e > 0).
            probe_dt, probe_t_max, probe_iters: sudden-stop probe step, horizon, and bisection
                iterations used to compute a_e_eff.
        """
        self.fwd = CarFollowingCBF1D(forward_spec, p_safe, gamma=gamma, L=L)
        self.rear = CoupledRearCBF(rear_model, rear_params, alpha1=alpha1, alpha2=alpha2,
                                   d_min=d_min, a_e=a_e, a_acc=a_acc,
                                   robust_factor=robust_factor)
        self.a_e = float(a_e)
        self.a_acc = float(a_acc)
        self.d_min = float(d_min)
        self.L = float(L)
        self.couple_a_e_eff = bool(couple_a_e_eff)
        self.a_e_eff_min = float(a_e_eff_min)
        self.probe_dt = float(probe_dt)
        self.probe_t_max = float(probe_t_max)
        self.probe_iters = int(probe_iters)
        # Assumed rear's own brake limit (caps a_r in the probe).
        self.rear_a_e_assumed = float(rear_params.get('a_e', a_e))
        # Diagnostics from the last filter() call.
        self.last_rhs_f = None
        self.last_lb_r = None
        self.last_a_e_eff = float(a_e)
        self.last_conflict = False
        self.last_status = 'none'

    def _probe_min_hr(self, a_e, h, v_ego, v_rear):
        """Min rear gap if the ego brakes at a constant `a_e` to rest, ASSUMED rear following.

        Forward-Euler sudden-stop rollout: ego decelerates at `a_e` (clamped at v=0) while the
        assumed rear reacts via its car-following law; returns the closest rear approach. Used
        to size the rear-aware braking authority a_e_eff.
        """
        dt = self.probe_dt
        worst = self.rear.robust_factor < 1.0
        ve, vr = float(v_ego), float(v_rear)
        h = float(h)
        min_h = h
        max_steps = int(self.probe_t_max / dt)
        for _ in range(max_steps):
            a_r = self.rear.rear_accel(h, ve, vr, worst=worst)
            a_r = max(a_r, -self.rear_a_e_assumed)
            h = h + (ve - vr) * dt                       # h_dot = v_ego - v_rear
            min_h = min(min_h, h)
            ve = max(ve - a_e * dt, 0.0)
            vr = max(vr + a_r * dt, 0.0)
            if ve <= 1e-6 and vr <= 1e-6:
                break
        return min_h

    def _a_e_eff(self, h, v_ego, v_rear):
        """Largest constant ego decel the ASSUMED rear survives (min h_r >= d_min); bisection."""
        lo, hi = self.a_e_eff_min, self.a_e
        if self._probe_min_hr(hi, h, v_ego, v_rear) >= self.d_min:
            return hi                                    # full authority is already survivable
        if self._probe_min_hr(lo, h, v_ego, v_rear) < self.d_min:
            return lo                                    # even the gentlest brake is unsafe
        for _ in range(self.probe_iters):
            mid = 0.5 * (lo + hi)
            if self._probe_min_hr(mid, h, v_ego, v_rear) >= self.d_min:
                lo = mid
            else:
                hi = mid
        return lo

    def filter(self, u_nom, x_ego, s_lead, v_lead, h_r, v_rear):
        """Safe ego acceleration closest to u_nom under both the forward and rear bounds.

        Args:
            u_nom: nominal (aggressive) ego acceleration.
            x_ego: ego state [s, v].
            s_lead, v_lead: lead position (center) and speed.
            h_r: current rear gap  h_r = s_ego - s_rear - L.
            v_rear: rear speed.
        """
        v_ego = float(np.array(x_ego).flatten()[1])

        # Rear-aware forward planning: how hard can the ego brake before the ASSUMED rear is
        # rear-ended? Feed that into the forward safe-distance ONLY (b_hat grows as a_e shrinks)
        # so the ego plans a larger gap and hangs back proactively. The actuator keeps full
        # braking authority (self.a_e) for genuine forward emergencies; rear protection is the
        # rear floor lb_r.
        if self.couple_a_e_eff:
            a_e_eff = self._a_e_eff(h_r, v_ego, v_rear)
            self.fwd.a_e = a_e_eff
            self.fwd.p['a_e'] = a_e_eff
        else:
            a_e_eff = self.a_e
        self.last_a_e_eff = float(a_e_eff)

        rhs_f = self.fwd.accel_upper_bound(x_ego, s_lead, v_lead)
        lb_r = self.rear.lower_bound(h_r, v_ego, v_rear)
        self.last_rhs_f, self.last_lb_r = float(rhs_f), float(lb_r)

        conflict = lb_r > rhs_f
        self.last_conflict = bool(conflict)
        if conflict:
            a = rhs_f                                   # forward-priority: never hit the lead
            self.last_status = 'conflict'
        else:
            a = min(max(u_nom, lb_r), rhs_f)            # rear floor, forward ceiling
            self.last_status = 'optimal'
        return float(np.clip(a, -self.a_e, self.a_acc))
