"""
1D double integrator robot class.

State: x = [s, v]  (position, velocity)
Dynamics: s_dot = v,  v_dot = u

Interface follows safe_control robot convention: f(X), g(X), step(X, U), df_dx(X).
"""

import numpy as np


class DoubleIntegrator1D:
    """
    1D double integrator: x = [s, v], x_dot = [v, u].

    All methods accept and return column vectors (2, 1).
    """

    def __init__(self, dt: float, robot_spec: dict):
        self.dt = dt
        self.robot_spec = robot_spec
        robot_spec.setdefault('model', 'DoubleIntegrator1D')
        robot_spec.setdefault('u_max', 1.0)
        robot_spec.setdefault('a_max', 1.0)
        robot_spec.setdefault('v_max', 10.0)

    def f(self, X) -> np.ndarray:
        """Drift term: f(x) = [v, 0]^T."""
        X = np.array(X).reshape(-1, 1)
        return np.array([[X[1, 0]], [0.0]])

    def g(self, X) -> np.ndarray:
        """Control matrix: g(x) = [0, 1]^T (constant)."""
        return np.array([[0.0], [1.0]])

    def step(self, X, U) -> np.ndarray:
        """Euler step: X_{k+1} = X_k + (f(X_k) + g(X_k) @ U_k) * dt."""
        X = np.array(X).reshape(-1, 1)
        U = np.array(U).reshape(-1, 1)
        return X + (self.f(X) + self.g(X) @ U) * self.dt

    def df_dx(self, X) -> np.ndarray:
        """Jacobian of f: A = [[0, 1], [0, 0]]."""
        return np.array([[0.0, 1.0], [0.0, 0.0]])
