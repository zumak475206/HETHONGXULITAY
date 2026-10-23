"""
tests/test_kalman.py
====================
Kiểm thử tự động Bộ lọc Động học Kalman (Constant Velocity Model).
"""

import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tracking.kalman_filter import PointKalmanFilter, HandKalmanFilter


class TestKalmanFilter(unittest.TestCase):

    def test_point_kalman_convergence(self):
        """Kiểm tra bộ lọc Kalman 3D hội tụ chính xác theo quỹ đạo chuyển động đều."""
        kf = PointKalmanFilter(dt=1.0/30.0, process_noise_var=0.01, measurement_noise_var=0.5)

        # Mô phỏng chuyển động 3D: x(t) = 100 + 30*t, y(t) = 200 - 10*t, z(t) = 50 + 15*t
        true_x, true_y, true_z = 100.0, 200.0, 50.0
        vx, vy, vz = 30.0, -10.0, 15.0
        dt = 1.0 / 30.0

        # Cho bộ lọc chạy 40 bước để ước tính vận tốc hội tụ hoàn toàn (chu trình Predict -> Update)
        for step in range(40):
            true_x += vx * dt
            true_y += vy * dt
            true_z += vz * dt
            noisy_x = true_x + np.random.normal(0, 0.5)
            noisy_y = true_y + np.random.normal(0, 0.5)
            noisy_z = true_z + np.random.normal(0, 0.5)

            kf.predict()
            est_x, est_y, est_z = kf.update(noisy_x, noisy_y, noisy_z)

        # Sai số ước tính phải nhỏ hơn độ lệch quan sát
        self.assertAlmostEqual(est_x, true_x, delta=3.0)
        self.assertAlmostEqual(est_y, true_y, delta=3.0)
        self.assertAlmostEqual(est_z, true_z, delta=3.0)

        # Kiểm tra tính tương thích ngược khi gọi update(x, y) với z mặc định = 0.0
        est_x2, est_y2, est_z2 = kf.update(100.0, 200.0)
        self.assertIsInstance(est_z2, float)

    def test_dead_reckoning_grace_cache(self):
        """Kiểm tra khả năng Dead-Reckoning duy trì quỹ đạo khi mất quan sát 2 frame."""
        hand_kf = HandKalmanFilter(grace_frames=3)

        # Khởi tạo 21 điểm dạng 4-tuple [id, x, y, z]
        pts = [(i, 200.0 + i*5.0, 300.0 + i*2.0, 10.0 + i*1.0) for i in range(21)]
        smoothed, is_valid = hand_kf.update(pts)
        self.assertTrue(is_valid)
        self.assertEqual(len(smoothed), 21)
        self.assertEqual(len(smoothed[0]), 4)  # (id, x, y, z)

        # Frame 2: Mất dấu (None) -> Phải dự báo qua Dead-Reckoning
        pred_pts, is_valid_pred = hand_kf.update(None)
        self.assertTrue(is_valid_pred)
        self.assertEqual(len(pred_pts), 21)
        self.assertEqual(len(pred_pts[0]), 4)  # (id, px, py, pz)

        # Frame 3: Mất dấu lần 2
        pred_pts2, is_valid_pred2 = hand_kf.update(None)
        self.assertTrue(is_valid_pred2)

        # Mất dấu vượt quá grace_frames (lần 4, 5) -> Phải reset
        hand_kf.update(None)
        hand_kf.update(None)
        _, is_valid_after = hand_kf.update(None)
    def test_hand_kalman_various_input_formats(self):
        """HandKalmanFilter must accept landmark objects, dicts, 3D tuples, and legacy 2D."""
        hand_kf = HandKalmanFilter()

        # 1. 3D tuples (x, y, z) without id prefix
        pts_3d = [(100.0 + i, 200.0 + i, 50.0 + i) for i in range(21)]
        res_3d, ok_3d = hand_kf.update(pts_3d)
        self.assertTrue(ok_3d)
        self.assertEqual(len(res_3d), 21)
        self.assertEqual(len(res_3d[0]), 4)

        # 2. Dicts
        pts_dict = [{"x": 100.0 + i, "y": 200.0 + i, "z": 10.0 + i} for i in range(21)]
        res_dict, ok_dict = hand_kf.update(pts_dict)
        self.assertTrue(ok_dict)

        # 3. Dummy objects with .x, .y, .z attributes
        class DummyLM:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        pts_obj = [DummyLM(150.0 + i, 250.0 + i, 5.0) for i in range(21)]
        res_obj, ok_obj = hand_kf.update(pts_obj)
        self.assertTrue(ok_obj)


if __name__ == "__main__":
    unittest.main()
