"""
Safe-distance surface for car following — the maximal invariant set of
He & Orosz, "Safety Guaranteed Connected Cruise Control" (ITSC 2018).

State (paper eq. 1):  [h, v, v1]
    h   bumper-to-bumper distance headway
    v   speed of the following (ego) vehicle,   v_dot = a   (control,    a  in [-a_e, a_bar])
    v1  speed of the preceding (lead) vehicle,  v1_dot = a1 (disturbance, a1 in [-a_l, a1_bar])
    h_dot = v1 - v

The maximal invariant (safety) set is (eq. 12)
    C = { (h, v, v1) | b(h, v, v1) = h - b_hat(v, v1) >= 0 },
where b_hat(v, v1) is the minimal safe distance for a minimum time headway tau.

b_hat is *continuous but not continuously differentiable* (kinks at the switching
curves).  `safe_distance` / `safe_distance_grad` give the exact value and a valid
(sub)gradient per active branch.  `SmoothSafeDistance` is a differentiable
over-approximation b_hat_s >= b_hat obtained by a polynomial regression fit plus a
constant upward shift; it is used as the terminal set of the Backup CBF.

Parameter dict `p` keys:
    tau    minimum time headway [s]
    a_e    ego maximum deceleration magnitude  (paper underline{a})   [m/s^2]
    a_l    lead maximum deceleration magnitude  (paper underline{a}_1) [m/s^2]
    vbar   maximum speed v_bar (used only for the smooth-fit grid)     [m/s]
"""

import numpy as np


# ----------------------------------------------------------------------
# Exact safe-distance surface b_hat(v, v1)  (eqs. 13-16)
# ----------------------------------------------------------------------

def safe_distance(v: float, v1: float, p: dict) -> float:
    """Minimal safe distance b_hat(v, v1) for minimum time headway tau.

    Branch selection follows eqs. 13-14 (a_e <= a_l) and 15-16 (a_e > a_l).
    The stopping-distance terms use max(v - a_e*tau, 0): once v <= a_e*tau the
    ego can stop within the time headway, so the extra term vanishes.
    """
    tau = float(p['tau'])
    a_e = float(p['a_e'])
    a_l = float(p['a_l'])

    v = float(v)
    v1 = float(v1)
    w = max(v - a_e * tau, 0.0)          # effective speed beyond the tau-stop budget

    if a_e <= a_l:
        # eq. 14 switching curve
        f1 = np.sqrt(a_l / a_e) * w
        if v1 >= f1:
            return v * tau
        # eq. 13 lower branch
        return v * tau + w * w / (2.0 * a_e) - v1 * v1 / (2.0 * a_l)

    # a_e > a_l : three branches (eqs. 15-16), f3 <= f2 since a_l/a_e < 1
    f2 = w
    f3 = (a_l / a_e) * w
    if v1 >= f2:
        return v * tau
    if v1 > f3:
        # both braking, relative-motion branch
        d = w - v1
        return v * tau + d * d / (2.0 * (a_e - a_l))
    # v1 <= f3 : independent stopping distances
    return v * tau + w * w / (2.0 * a_e) - v1 * v1 / (2.0 * a_l)


def safe_distance_grad(v: float, v1: float, p: dict) -> tuple:
    """Analytic (sub)gradient (d b_hat / d v, d b_hat / d v1) for the active branch."""
    tau = float(p['tau'])
    a_e = float(p['a_e'])
    a_l = float(p['a_l'])

    v = float(v)
    v1 = float(v1)
    w = max(v - a_e * tau, 0.0)
    dw_dv = 1.0 if (v - a_e * tau) > 0.0 else 0.0     # d w / d v

    if a_e <= a_l:
        f1 = np.sqrt(a_l / a_e) * w
        if v1 >= f1:
            return tau, 0.0
        db_dv = tau + (w / a_e) * dw_dv
        db_dv1 = -v1 / a_l
        return db_dv, db_dv1

    f2 = w
    f3 = (a_l / a_e) * w
    if v1 >= f2:
        return tau, 0.0
    if v1 > f3:
        d = w - v1
        db_dv = tau + (d / (a_e - a_l)) * dw_dv
        db_dv1 = -d / (a_e - a_l)
        return db_dv, db_dv1
    db_dv = tau + (w / a_e) * dw_dv
    db_dv1 = -v1 / a_l
    return db_dv, db_dv1


# ----------------------------------------------------------------------
# Smooth over-approximation b_hat_s(v, v1) >= b_hat  (regression fit)
# ----------------------------------------------------------------------

class SmoothSafeDistance:
    """Differentiable polynomial fit to b_hat, shifted up so that b_hat_s >= b_hat.

    A bivariate polynomial of total degree `degree` is least-squares fit to b_hat
    sampled on an (n_grid x n_grid) grid over [0, vbar]^2.  The maximum grid
    under-prediction max(b_hat - fit, 0) is then added as a constant offset, which
    guarantees b_hat_s >= b_hat on the sample grid (and, with a fine grid, in
    between to within the fit's smoothness).
    """

    def __init__(self, p: dict, degree: int = 4, n_grid: int = 60,
                 margin: float = 0.0):
        self.p = dict(p)
        self.degree = int(degree)
        vbar = float(p.get('vbar', 12.0))

        # Monomial exponent list for total degree <= degree.
        self._terms = [(i, j) for i in range(self.degree + 1)
                       for j in range(self.degree + 1 - i)]

        vs = np.linspace(0.0, vbar, n_grid)
        v1s = np.linspace(0.0, vbar, n_grid)
        VV, VV1 = np.meshgrid(vs, v1s, indexing='ij')
        B = np.vectorize(lambda a, b: safe_distance(a, b, p))(VV, VV1)

        A = self._design(VV.ravel(), VV1.ravel())          # (n_samples, n_terms)
        coef, *_ = np.linalg.lstsq(A, B.ravel(), rcond=None)
        self.coef = coef

        fit = (A @ coef).reshape(VV.shape)
        under = np.max(np.maximum(B - fit, 0.0))           # worst under-prediction
        self.offset = float(under + margin)
        self.max_fit_error = float(np.max(np.abs(B - fit)))

    # -- internal: design matrix and its derivatives -----------------------

    def _design(self, v, v1):
        v = np.asarray(v, dtype=float)
        v1 = np.asarray(v1, dtype=float)
        return np.stack([v ** i * v1 ** j for (i, j) in self._terms], axis=-1)

    def value(self, v: float, v1: float) -> float:
        cols = np.array([v ** i * v1 ** j for (i, j) in self._terms])
        return float(cols @ self.coef + self.offset)

    def grad(self, v: float, v1: float) -> tuple:
        dv = np.array([(i * v ** (i - 1) if i > 0 else 0.0) * v1 ** j
                       for (i, j) in self._terms])
        dv1 = np.array([v ** i * (j * v1 ** (j - 1) if j > 0 else 0.0)
                        for (i, j) in self._terms])
        return float(dv @ self.coef), float(dv1 @ self.coef)


# ----------------------------------------------------------------------
# Self-test: continuity across the switch and b_hat_s >= b_hat
# ----------------------------------------------------------------------

def _self_test() -> None:
    import itertools

    for a_e, a_l in [(3.0, 4.0), (4.0, 3.0), (3.0, 3.0)]:
        p = {'tau': 1.0, 'a_e': a_e, 'a_l': a_l, 'vbar': 12.0}

        # Continuity across the switching curve(s): compare across a fine grid.
        vs = np.linspace(0.0, 12.0, 200)
        v1s = np.linspace(0.0, 12.0, 200)
        max_jump = 0.0
        for v in vs:
            row = np.array([safe_distance(v, v1, p) for v1 in v1s])
            max_jump = max(max_jump, float(np.max(np.abs(np.diff(row)))))
        dv1 = v1s[1] - v1s[0]
        assert max_jump < 5.0 * dv1 + 1e-6, \
            f"b_hat discontinuous (a_e={a_e}, a_l={a_l}): max step {max_jump:.4f}"

        # Smooth fit upper-bounds b_hat on a validation grid.
        smooth = SmoothSafeDistance(p, degree=4, n_grid=60)
        worst = 0.0
        for v, v1 in itertools.product(np.linspace(0, 12, 80),
                                       np.linspace(0, 12, 80)):
            worst = min(worst, smooth.value(v, v1) - safe_distance(v, v1, p))
        assert worst >= -1e-6, \
            f"b_hat_s < b_hat (a_e={a_e}, a_l={a_l}): min margin {worst:.4f}"

        print(f"a_e={a_e}, a_l={a_l}:  max b_hat step={max_jump:.4f} m, "
              f"fit RMS-ish err={smooth.max_fit_error:.3f} m, "
              f"offset={smooth.offset:.3f} m, min(b_hat_s - b_hat)={worst:.4f} m  OK")

    print("self-test passed")


if __name__ == '__main__':
    _self_test()
