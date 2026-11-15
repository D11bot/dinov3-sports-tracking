"""I use a constant velocity Kalman filter to predict player positions."""
import numpy as np

import numpy as np

class CVKalman:
    """Constant-velocity Kalman filter for box center."""

    def __init__(self, box):
        x, y, w, h = map(float, box)

        self.x = np.array([
            x + w / 2,
            y + h / 2,
            0.0,
            0.0
        ])

        self.P = np.diag([25., 25., 100., 100.])

        self.F = np.array([
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])

        self.H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])

        self.Q = np.diag([4., 4., 9., 9.])
        self.R = np.diag([16., 16.])

        self.w = w
        self.h = h

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, box):
        bx, by, bw, bh = map(float, box)

        z = np.array([
            bx + bw / 2,
            by + bh / 2
        ])

        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        d2 = float(
            residual @ np.linalg.solve(S, residual)
        )

        return residual, S, d2

    def update(self, box):
        residual, S, _ = self.innovation(box)

        K = np.linalg.solve(
            S, self.H @ self.P
        ).T

        self.x = self.x + K @ residual

        I = np.eye(4)
        A = I - K @ self.H

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        self.w = float(box[2])
        self.h = float(box[3])

    def predicted_box(self):
        return np.array([
            self.x[0] - self.w / 2,
            self.x[1] - self.h / 2,
            self.w,
            self.h
        ])

class CalibratedCVKalman(CVKalman):
    def __init__(self, box, r_variance=16.0):
        super().__init__(box)

        self.R = np.eye(2, dtype=float) * float(r_variance)
