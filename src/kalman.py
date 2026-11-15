"""I use a constant velocity Kalman filter to predict player positions."""

import numpy as np


class CVKalman:
    """I track the box center with the state (x, y, vx, vy).

    I keep the last measured box width and height for the predicted box.
    I set the measurement noise R with r_variance in px^2, the same for x and y.
    """

    def __init__(self, box, r_variance=16.0):
        x, y, w, h = map(float, box)

        self.x = np.array([x + w / 2, y + h / 2, 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 100.0, 100.0])

        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        self.Q = np.diag([4.0, 4.0, 9.0, 9.0])
        self.R = np.eye(2) * float(r_variance)

        self.w = w
        self.h = h

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, box):
        """I return the residual, the innovation covariance S and the squared Mahalanobis distance d2."""
        bx, by, bw, bh = map(float, box)
        z = np.array([bx + bw / 2, by + bh / 2])

        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float(residual @ np.linalg.solve(S, residual))

        return residual, S, d2

    def update(self, box):
        residual, S, _ = self.innovation(box)
        K = np.linalg.solve(S, self.H @ self.P).T

        self.x = self.x + K @ residual

        # I use the Joseph form so P stays symmetric and numerically stable.
        A = np.eye(4) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T

        self.w = float(box[2])
        self.h = float(box[3])

    def predicted_box(self):
        return np.array(
            [
                self.x[0] - self.w / 2,
                self.x[1] - self.h / 2,
                self.w,
                self.h,
            ]
        )
