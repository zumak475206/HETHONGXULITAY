"""
src/tracking/kalman_filter.py
Kalman filter upgraded: 4D [x,y,vx,vy] -> 6D [x,y,z,vx,vy,vz]

State vector: x = [px, py, pz, vx, vy, vz]^T
Transition F 6x6: constant velocity model extended to 3D
Observation H 3x6: camera now observes (px,py,pz) from MediaPipe
Q, R, P updated to 6D/3D accordingly.
Dead-Reckoning: predict-only when hand lost for up to grace_frames.
"""

import numpy as np
from typing import List, Tuple, Optional, Any


class PointKalmanFilter:
    """Kalman filter for one 3D point (px, py, pz)."""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_var: float = 0.05,
        measurement_noise_var: float = 0.80,
        error_cov_post: float = 1.0,
    ):
        self.dt = float(dt)
        self.q_var = float(process_noise_var)
        self.r_var = float(measurement_noise_var)

        # State: [x, y, z, vx, vy, vz]^T  (6D)
        self.x = np.zeros((6, 1), dtype=np.float64)

        # Transition matrix F (6x6) - Constant Velocity in 3D
        dt = self.dt
        self.F = np.array([
            [1.0, 0.0, 0.0, dt,  0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt,  0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt ],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Observation matrix H (3x6): observe (x, y, z)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Process noise Q (6x6) - Discrete Constant White Noise Acceleration Model
        # G = [dt^2/2, dt]^T -> Q_block = G * G^T * q_var = [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * q_var
        dt2 = self.dt ** 2
        dt3 = (self.dt ** 3) / 2.0
        dt4 = (self.dt ** 4) / 4.0
        self.Q = np.zeros((6, 6), dtype=np.float64)
        for s, v in [(0, 3), (1, 4), (2, 5)]:
            self.Q[s, s] = dt4
            self.Q[s, v] = dt3
            self.Q[v, s] = dt3
            self.Q[v, v] = dt2
        self.Q *= self.q_var

        # Measurement noise R (3x3)
        self.R = np.eye(3, dtype=np.float64) * self.r_var

        # Error covariance P (6x6)
        self.P = np.eye(6, dtype=np.float64) * float(error_cov_post)

        self.initialized = False
        self.lost_frames = 0

    def init_state(self, px: float, py: float, pz: float = 0.0):
        """Initialize state from first observation."""
        self.x = np.array([[float(px)], [float(py)], [float(pz)],
                           [0.0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = True
        self.lost_frames = 0

    def predict(self) -> Tuple[float, float, float]:
        """Prediction step: x = F*x, P = F*P*F^T + Q."""
        if not self.initialized:
            return 0.0, 0.0, 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

    def update(self, px: float, py: float, pz: float = 0.0) -> Tuple[float, float, float]:
        """Update step with new observation z = [px, py, pz]^T."""
        if not self.initialized:
            self.init_state(px, py, pz)
            return px, py, pz
        z = np.array([[float(px)], [float(py)], [float(pz)]], dtype=np.float64)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(6, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        self.lost_frames = 0
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

    def get_velocity(self) -> Tuple[float, float, float]:
        """Return estimated velocity (vx, vy, vz)."""
        return float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0])


class HandKalmanFilter:
    """Bank of 21 PointKalmanFilter3D for all hand landmarks with dead-reckoning."""

    def __init__(
        self,
        grace_frames: int = 3,
        dt: float = 1.0 / 30.0,
        process_noise_var: float = 0.05,
        measurement_noise_var: float = 0.80,
    ):
        self.grace_frames = grace_frames
        self.filters = [
            PointKalmanFilter(
                dt=dt,
                process_noise_var=process_noise_var,
                measurement_noise_var=measurement_noise_var,
            )
            for _ in range(21)
        ]
        self.has_active_hand = False
        self.lost_frames_counter = 0

    def update(
        self, lm_list: Optional[List[Any]]
    ) -> Tuple[List[Tuple[int, float, float, float]], bool]:
        """Process raw landmarks. Returns (smoothed_list [(id,x,y,z),...], is_valid).

        Accepts both old [id,x,y] format (z=0) and new [id,x,y,z] format.
        """
        if lm_list is not None and len(lm_list) >= 21:
            smoothed_list = []
            for i in range(21):
                raw_pt = lm_list[i]
                if hasattr(raw_pt, "x") and hasattr(raw_pt, "y"):
                    rx, ry, rz = float(raw_pt.x), float(raw_pt.y), float(getattr(raw_pt, "z", 0.0))
                elif isinstance(raw_pt, dict):
                    rx, ry, rz = float(raw_pt.get("x", 0.0)), float(raw_pt.get("y", 0.0)), float(raw_pt.get("z", 0.0))
                elif len(raw_pt) >= 4:
                    rx, ry, rz = float(raw_pt[1]), float(raw_pt[2]), float(raw_pt[3])
                elif len(raw_pt) == 3:
                    try:
                        idx = int(raw_pt[0])
                        if 0 <= idx <= 20 and isinstance(raw_pt[0], (int, np.integer)):
                            rx, ry, rz = float(raw_pt[1]), float(raw_pt[2]), 0.0
                        else:
                            rx, ry, rz = float(raw_pt[0]), float(raw_pt[1]), float(raw_pt[2])
                    except Exception:
                        rx, ry, rz = float(raw_pt[0]), float(raw_pt[1]), float(raw_pt[2])
                elif len(raw_pt) == 2:
                    rx, ry, rz = float(raw_pt[0]), float(raw_pt[1]), 0.0
                else:
                    rx, ry, rz = 0.0, 0.0, 0.0
                self.filters[i].predict()
                sx, sy, sz = self.filters[i].update(rx, ry, rz)
                smoothed_list.append((i, sx, sy, sz))
            self.has_active_hand = True
            self.lost_frames_counter = 0
            return smoothed_list, True

        # Dead-Reckoning when lost
        if self.has_active_hand and self.lost_frames_counter < self.grace_frames:
            self.lost_frames_counter += 1
            predicted_list = []
            for i in range(21):
                px, py, pz = self.filters[i].predict()
                predicted_list.append((i, px, py, pz))
            return predicted_list, True

        self.reset()
        return [], False

    def get_hand_velocity(self) -> Tuple[float, float, float]:
        """Return wrist velocity (vx, vy, vz)."""
        if self.has_active_hand:
            return self.filters[0].get_velocity()
        return 0.0, 0.0, 0.0

    def reset(self):
        """Reset all 21 filters."""
        self.has_active_hand = False
        self.lost_frames_counter = 0
        for f in self.filters:
            f.initialized = False
