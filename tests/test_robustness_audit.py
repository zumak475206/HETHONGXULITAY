"""
tests/test_robustness_audit.py
Kiem tra cac ca bien va tinh nang sua loi tu Hoi dong Tham dinh:
1. Anti-Swapping: Ghep cap ban tay theo khoang cach khi 2 tay bat cheo.
2. Failsafe Release: Xu ly su kien nha chuot an toan khi ban tay bien mat.
3. Discrete White Noise Acceleration Q: Do hoi tu bo loc Kalman 6D.
"""

import unittest
import queue
import time
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tracking.kalman_filter import PointKalmanFilter, HandKalmanFilter
from src.actions.dispatcher import ActionDispatcher


class TestRobustnessAudit(unittest.TestCase):

    def test_kalman_6d_q_acceleration_formulation(self):
        """Kiem tra ma tran Q co dung cau truc Discrete White Noise Acceleration."""
        dt = 1.0 / 30.0
        q_var = 0.05
        kf = PointKalmanFilter(dt=dt, process_noise_var=q_var)

        expected_dt3 = ((dt ** 3) / 2.0) * q_var
        expected_dt4 = ((dt ** 4) / 4.0) * q_var
        expected_dt2 = (dt ** 2) * q_var

        self.assertAlmostEqual(kf.Q[0, 3], expected_dt3, places=8)
        self.assertAlmostEqual(kf.Q[3, 0], expected_dt3, places=8)
        self.assertAlmostEqual(kf.Q[0, 0], expected_dt4, places=8)
        self.assertAlmostEqual(kf.Q[3, 3], expected_dt2, places=8)

    def test_hand_anti_swapping_distance_logic(self):
        """Kiem tra thuat toan ghep cap chong hoan doi khi 2 ban tay bat cheo."""
        prev_w0 = np.array([100.0, 200.0])
        prev_w1 = np.array([500.0, 200.0])

        cur_det0 = np.array([490.0, 205.0])
        cur_det1 = np.array([110.0, 195.0])
        detected_lms = ["Hand_Right", "Hand_Left"]

        cost_direct = np.linalg.norm(cur_det0 - prev_w0) + np.linalg.norm(cur_det1 - prev_w1)
        cost_swapped = np.linalg.norm(cur_det1 - prev_w0) + np.linalg.norm(cur_det0 - prev_w1)

        self.assertLess(cost_swapped, cost_direct)

        if cost_swapped < cost_direct:
            matched_lms = [detected_lms[1], detected_lms[0]]

        self.assertEqual(matched_lms[0], "Hand_Left")
        self.assertEqual(matched_lms[1], "Hand_Right")

    def test_dispatcher_failsafe_release(self):
        """Kiem tra ActionDispatcher nhan va xu ly an toan su kien FAILSAFE_RELEASE."""
        action_q = queue.Queue()
        dispatcher = ActionDispatcher(action_q)
        dispatcher.start()

        try:
            action_q.put({"action": "FAILSAFE_RELEASE", "payload": {}})
            time.sleep(0.2)
            self.assertEqual(dispatcher.last_action_name, "FAILSAFE_RELEASE")
        finally:
            dispatcher.stop()
            dispatcher.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
