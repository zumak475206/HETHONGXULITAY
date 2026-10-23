"""
tests/test_voting.py
====================
Kiểm thử tự động Bộ lọc 5-Frame Majority Voting và Cooldown State Machine.
"""

import unittest
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.voting_smoother import VotingSmoother


class TestVotingSmoother(unittest.TestCase):

    def test_majority_voting_and_cooldown(self):
        """Kiểm tra vote đa số 5-frame và khóa đệm cooldown chống kích hoạt lặp."""
        smoother = VotingSmoother(window_size=5, min_ratio=0.60, confidence_threshold=0.80, cooldown_sec=0.4)

        # 4 frame đầu: chưa đủ cửa sổ 5 frame
        for _ in range(4):
            act, _, _ = smoother.update("SWIPE_LEFT", 0.95)
            self.assertIsNone(act)

        # Frame thứ 5: Đủ 5 frame đồng thuận -> Kích hoạt lần 1
        act, conf, stable = smoother.update("SWIPE_LEFT", 0.95)
        self.assertEqual(act, "SWIPE_LEFT")
        self.assertEqual(stable, "SWIPE_LEFT")

        # Các frame tiếp theo khi giữ nguyên tay -> Phải bị khóa HOLDING (chống auto-repeat)
        for _ in range(3):
            act_locked, _, stable_l = smoother.update("SWIPE_LEFT", 0.95)
            self.assertIsNone(act_locked)
            self.assertTrue("HOLDING" in stable_l or "LOCKOUT" in stable_l)

        # Chờ hết cooldown
        time.sleep(0.45)

        # Nếu vẫn giữ nguyên tay -> Vẫn bị khóa HOLDING (không tự động bắn lặp lại)
        act_still_holding, _, _ = smoother.update("SWIPE_LEFT", 0.95)
        self.assertIsNone(act_still_holding)

        # Hạ tay về IDLE -> Hệ thống Re-arm Edge-Trigger
        for _ in range(5):
            smoother.update("IDLE", 0.99)

        # Đưa tay lên thực hiện lại -> Kích hoạt thành công lần 2
        # Khi buffer đã đầy (toàn IDLE), chỉ cần 3/5 frame (>= 60% min_ratio) là đạt đa số
        act_f1, _, _ = smoother.update("SWIPE_LEFT", 0.95)
        self.assertIsNone(act_f1)  # 1/5 = 20%
        act_f2, _, _ = smoother.update("SWIPE_LEFT", 0.95)
        self.assertIsNone(act_f2)  # 2/5 = 40%
        act_f3, _, stable_f3 = smoother.update("SWIPE_LEFT", 0.95)
        self.assertEqual(act_f3, "SWIPE_LEFT")  # 3/5 = 60% -> Đạt ngưỡng đa số, kích hoạt ngay lập tức
        self.assertEqual(stable_f3, "SWIPE_LEFT")

        # Các frame tiếp theo tiếp tục bị khóa HOLDING chống auto-repeat
        act_f4, _, _ = smoother.update("SWIPE_LEFT", 0.95)
        self.assertIsNone(act_f4)
        act_f5, _, _ = smoother.update("SWIPE_LEFT", 0.95)
        self.assertIsNone(act_f5)

    def test_low_confidence_rejection(self):
        """Kiểm tra độ tin cậy thấp (< 80%) bị từ chối không kích hoạt."""
        smoother = VotingSmoother(window_size=5, confidence_threshold=0.80)
        for _ in range(10):
            act, _, stable = smoother.update("PINCH_CLICK", 0.50)  # Thấp hơn 80%
            self.assertIsNone(act)


if __name__ == "__main__":
    unittest.main()
