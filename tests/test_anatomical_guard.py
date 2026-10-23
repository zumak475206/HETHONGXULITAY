"""
tests/test_anatomical_guard.py
==============================
Unit test kiểm thử Bộ lọc giải phẫu học sinh học (Anatomical Sanity Guard).
Đảm bảo chặn đứng 100% hiện tượng ảo giác kéo xương bàn tay lên bờ vai/ngực/mặt.
"""

import unittest
import math
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.tracking.anatomical_guard import AnatomicalGuard


def create_realistic_hand(wrist=(300.0, 350.0), scale=100.0) -> list:
    """
    Tạo bộ 21 landmark bàn tay tự nhiên chuẩn giải phẫu người.
    Kích thước lòng bàn tay (WRIST -> MIDDLE_MCP) xấp xỉ `scale` pixel.
    """
    wx, wy = wrist
    # WRIST = 0
    pts = [(0, wx, wy, 0.0)]

    # THUMB (1..4)
    pts.append((1, wx - 0.25 * scale, wy - 0.20 * scale, 0.0))  # CMC
    pts.append((2, wx - 0.40 * scale, wy - 0.45 * scale, 0.0))  # MCP
    pts.append((3, wx - 0.50 * scale, wy - 0.70 * scale, 0.0))  # IP
    pts.append((4, wx - 0.55 * scale, wy - 0.90 * scale, 0.0))  # TIP

    # INDEX (5..8)
    pts.append((5, wx - 0.30 * scale, wy - 0.95 * scale, 0.0))  # MCP
    pts.append((6, wx - 0.32 * scale, wy - 1.35 * scale, 0.0))  # PIP (proximal ~0.40*scale)
    pts.append((7, wx - 0.33 * scale, wy - 1.62 * scale, 0.0))  # DIP (intermediate ~0.27*scale)
    pts.append((8, wx - 0.34 * scale, wy - 1.82 * scale, 0.0))  # TIP (distal ~0.20*scale)

    # MIDDLE (9..12)
    pts.append((9,  wx,                 wy - 1.00 * scale, 0.0))  # MCP (palm_size = 1.0*scale)
    pts.append((10, wx,                 wy - 1.45 * scale, 0.0))  # PIP (~0.45*scale)
    pts.append((11, wx,                 wy - 1.75 * scale, 0.0))  # DIP (~0.30*scale)
    pts.append((12, wx,                 wy - 1.98 * scale, 0.0))  # TIP (~0.23*scale)

    # RING (13..16)
    pts.append((13, wx + 0.25 * scale, wy - 0.95 * scale, 0.0))  # MCP
    pts.append((14, wx + 0.27 * scale, wy - 1.35 * scale, 0.0))  # PIP
    pts.append((15, wx + 0.28 * scale, wy - 1.63 * scale, 0.0))  # DIP
    pts.append((16, wx + 0.29 * scale, wy - 1.84 * scale, 0.0))  # TIP

    # PINKY (17..20)
    pts.append((17, wx + 0.45 * scale, wy - 0.85 * scale, 0.0))  # MCP
    pts.append((18, wx + 0.50 * scale, wy - 1.18 * scale, 0.0))  # PIP
    pts.append((19, wx + 0.52 * scale, wy - 1.40 * scale, 0.0))  # DIP
    pts.append((20, wx + 0.54 * scale, wy - 1.58 * scale, 0.0))  # TIP

    return pts


class TestAnatomicalGuard(unittest.TestCase):

    def setUp(self):
        self.guard = AnatomicalGuard(
            enabled=True,
            max_finger_palm_ratio=1.25,
            max_segment_palm_ratio=0.85,
            min_phalanx_ratio=0.35,
            max_phalanx_ratio=3.50,
            min_palm_size=10.0,
        )

    def test_normal_hand_valid(self):
        """Bàn tay chuẩn bình thường phải vượt qua bộ lọc 100%."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        is_valid, reason = self.guard.validate_hand(hand)
        self.assertTrue(is_valid, f"Bàn tay bình thường bị từ chối: {reason}")
        self.assertTrue(self.guard.is_valid_hand(hand))

    def test_hallucination_to_shoulder_blocked(self):
        """Mô phỏng ảnh thực tế: Ngón trỏ bị kéo dãn tít lên đỉnh vai/mặt -> Phải chặn đứng 100%."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Kéo đầu ngón trỏ (8) và khớp DIP (7) bay lên góc vai (100, 50)
        hand[7] = (7, 150.0, 100.0, 0.0)
        hand[8] = (8, 100.0, 50.0, 0.0)

        is_valid, reason = self.guard.validate_hand(hand)
        self.assertFalse(is_valid, "Ảo giác kéo lên vai không bị phát hiện!")
        self.assertIn("index", reason.lower())

    def test_single_segment_too_long_blocked(self):
        """Một đốt xương đơn lẻ dài bất thường (> 0.85 palm_size) -> Bị chặn."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Kéo PIP ngón giữa (10) vọt lên xa
        hand[10] = (10, hand[10][1], hand[10][2] - 100.0, 0.0)
        is_valid, reason = self.guard.validate_hand(hand)
        self.assertFalse(is_valid)

    def test_phalanx_ratio_distortion_blocked(self):
        """Tỷ lệ giữa các đốt xương bị méo mó nghiêm trọng (phá vỡ tỷ lệ vàng) -> Bị chặn."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Làm đốt giữa ngón trỏ (PIP->DIP) dài gấp 5 lần đốt gần (MCP->PIP)
        mcp = hand[5]
        # Thu nhỏ đốt gần
        hand[6] = (6, mcp[1], mcp[2] - 5.0, 0.0)
        # Kéo dãn đốt giữa
        hand[7] = (7, mcp[1], mcp[2] - 80.0, 0.0)

        is_valid, reason = self.guard.validate_hand(hand)
        self.assertFalse(is_valid)

    def test_thumb_hallucination_blocked(self):
        """Ngón cái bị kéo dãn bất thường -> Bị chặn."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        hand[4] = (4, hand[4][1] - 150.0, hand[4][2] - 150.0, 0.0)

        is_valid, reason = self.guard.validate_hand(hand)
        self.assertFalse(is_valid)

    def test_small_palm_size_rejected(self):
        """Kích thước bàn tay quá nhỏ (< 10px) -> Bị từ chối."""
        tiny_hand = create_realistic_hand(wrist=(50.0, 50.0), scale=5.0)
        is_valid, reason = self.guard.validate_hand(tiny_hand)
        self.assertFalse(is_valid)
        self.assertIn("too small", reason.lower())

    def test_empty_or_incomplete_landmarks(self):
        """Dữ liệu rỗng hoặc thiếu điểm (< 21 điểm) -> Bị từ chối an toàn."""
        self.assertFalse(self.guard.validate_hand([])[0])
        self.assertFalse(self.guard.validate_hand(None)[0])
        self.assertFalse(self.guard.validate_hand([(i, 0, 0) for i in range(15)])[0])

    def test_nan_or_inf_coordinates(self):
        """Tọa độ chứa NaN hoặc Inf -> Bị từ chối an toàn."""
        hand = create_realistic_hand()
        hand[8] = (8, float("nan"), 100.0, 0.0)
        self.assertFalse(self.guard.validate_hand(hand)[0])

        hand2 = create_realistic_hand()
        hand2[8] = (8, float("inf"), 100.0, 0.0)
        self.assertFalse(self.guard.validate_hand(hand2)[0])

    def test_different_landmark_formats(self):
        """Hỗ trợ nhiều định dạng: tuple 4D, tuple 3D, tuple 2D, dict."""
        hand4d = create_realistic_hand()
        self.assertTrue(self.guard.validate_hand(hand4d)[0])

        # Format 3D: (id, x, y)
        hand3d = [(pt[0], pt[1], pt[2]) for pt in hand4d]
        self.assertTrue(self.guard.validate_hand(hand3d)[0])

        # Format 2D: (x, y)
        hand2d = [(pt[1], pt[2]) for pt in hand4d]
        self.assertTrue(self.guard.validate_hand(hand2d)[0])

        # Format dict: [{'x': ..., 'y': ...}]
        hand_dict = [{"x": pt[1], "y": pt[2]} for pt in hand4d]
        self.assertTrue(self.guard.validate_hand(hand_dict)[0])

    def test_wrist_hallucination_to_shoulder_blocked(self):
        """Mô phỏng cổ tay (wrist=0) bị kéo dãn tít lên vai/ngực -> Phải chặn đứng 100%."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Kéo cổ tay vọt lên (100.0, 50.0) cách xa các khớp MCP
        hand[0] = (0, 100.0, 50.0, 0.0)
        is_valid, reason = self.guard.validate_hand(hand)
        self.assertFalse(is_valid, "Ảo giác kéo cổ tay lên vai không bị phát hiện!")

    def test_palm_aspect_ratio_distortion_blocked(self):
        """Mô phỏng bàn tay bị méo dạng bất thường: khoảng cách cổ tay - khớp MCP quá dài so với bề ngang bàn tay."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Kéo cổ tay xuống rất xa làm palm aspect ratio > 2.20
        hand[0] = (0, 320.0, 600.0, 0.0)
        is_valid, reason = self.guard.validate_hand(hand)
        self.assertFalse(is_valid, "Tỷ lệ dài/rộng lòng bàn tay méo mó không bị phát hiện!")

    def test_coordinates_starting_with_small_integer(self):
        """Kiểm tra format (x, y, z) khi x vô tình là số thực nhỏ <= 20 (ví dụ mép màn hình x=15.0)."""
        hand = create_realistic_hand(wrist=(15.0, 300.0), scale=100.0)
        # Format (x, y, z) không có landmark ID
        hand_xyz = [(pt[1], pt[2], pt[3]) for pt in hand]
        # Điểm đầu tiên x=15.0 (trùng range 0..20 của ID nhưng không phải ID)
        is_valid, reason = self.guard.validate_hand(hand_xyz)
        self.assertTrue(is_valid, f"Format (x, y, z) với x=15 bị parse nhầm thành landmark ID: {reason}")


if __name__ == "__main__":
    unittest.main()
