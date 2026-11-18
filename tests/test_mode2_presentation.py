"""
tests/test_mode2_presentation.py
=================================
Unit tests kiểm thử toàn diện Chế độ 2: Trình chiếu Slide (Presentation Mode).
- Nhận diện phím nóng Chữ L (L-Sign Hold 1.0s -> F5).
- Quy trình Chuyển Slide (Giơ tay 1s để nạp -> Quơ trái/phải).
- Chống kích hoạt lặp bằng Cooldown 0.8s.
- Hủy nạp khi hạ tay hoặc nắm tay.
"""

import unittest
from unittest.mock import patch, MagicMock
import time
import math
import queue
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.modes.mode2_presentation import PresentationController
from src.actions.dispatcher import ActionDispatcher


def make_realistic_hand_pts(wrist=(320.0, 300.0), scale=100.0, fingers_up=(1, 1, 1, 1, 1)):
    """
    Tạo danh sách 21 điểm landmark giả lập bàn tay thực tế:
    fingers_up: tuple (thumb, index, middle, ring, pinky) với 1=duỗi, 0=gập.
    """
    wx, wy = wrist
    pts = [(i, wx, wy, 0.0) for i in range(21)]

    # MCP joints (gốc các ngón)
    pts[config.MIDDLE_MCP] = (config.MIDDLE_MCP, wx, wy - scale, 0.0)
    pts[config.INDEX_MCP]  = (config.INDEX_MCP, wx - scale * 0.35, wy - scale * 0.9, 0.0)
    pts[config.RING_MCP]   = (config.RING_MCP, wx + scale * 0.30, wy - scale * 0.9, 0.0)
    pts[config.PINKY_MCP]  = (config.PINKY_MCP, wx + scale * 0.60, wy - scale * 0.75, 0.0)
    pts[config.THUMB_MCP]  = (config.THUMB_MCP, wx - scale * 0.45, wy - scale * 0.35, 0.0)
    pts[config.THUMB_IP]   = (config.THUMB_IP, wx - scale * 0.65, wy - scale * 0.50, 0.0)

    # Cài đặt trạng thái từng ngón
    # 1. Ngón cái
    if fingers_up[0]:
        # Ngón cái duỗi xòe ra ngoài (xa ngón trỏ)
        pts[config.THUMB_TIP] = (config.THUMB_TIP, wx - scale * 0.85, wy - scale * 0.65, 0.0)
    else:
        # Ngón cái gập áp sát vào lòng bàn tay
        pts[config.THUMB_TIP] = (config.THUMB_TIP, wx - scale * 0.20, wy - scale * 0.60, 0.0)

    # 4 ngón còn lại
    finger_indices = [
        (config.INDEX_MCP, config.INDEX_PIP, config.INDEX_DIP, config.INDEX_TIP, -0.35),
        (config.MIDDLE_MCP, config.MIDDLE_PIP, config.MIDDLE_DIP, config.MIDDLE_TIP, 0.0),
        (config.RING_MCP, config.RING_PIP, config.RING_DIP, config.RING_TIP, 0.30),
        (config.PINKY_MCP, config.PINKY_PIP, config.PINKY_DIP, config.PINKY_TIP, 0.60),
    ]

    for f_idx, (mcp, pip, dip, tip, offset_x) in enumerate(finger_indices):
        is_up = fingers_up[f_idx + 1]
        base_x = wx + scale * offset_x
        base_y = wy - scale * 0.9
        if is_up:
            # Ngón duỗi thẳng lên trên (Y giảm)
            pts[pip] = (pip, base_x, base_y - scale * 0.35, 0.0)
            pts[dip] = (dip, base_x, base_y - scale * 0.65, 0.0)
            pts[tip] = (tip, base_x, base_y - scale * 0.95, 0.0)
        else:
            # Ngón gập cuộn tròn lại xuống dưới
            pts[pip] = (pip, base_x, base_y - scale * 0.20, 0.0)
            pts[dip] = (dip, base_x, base_y + scale * 0.10, 0.0)
            pts[tip] = (tip, base_x, base_y + scale * 0.30, 0.0)

    return pts


class TestMode2Presentation(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.action_queue = queue.Queue()
        self.controller = PresentationController(
            frame_w=1280,
            frame_h=720,
            action_queue=self.action_queue,
        )

    def test_l_sign_detection_and_f5_trigger(self):
        """
        Kiểm tra phím nóng chữ L:
        - Giơ ngón cái + ngón trỏ duỗi (L-sign), 3 ngón còn lại gập.
        - Giữ 1.0 giây -> Bắn lệnh F5 (PRESENTATION_START).
        """
        # Tạo bàn tay chữ L: (1, 1, 0, 0, 0)
        hand_l = make_realistic_hand_pts(wrist=(640.0, 450.0), scale=120.0, fingers_up=(1, 1, 0, 0, 0))

        # Frame 1: Bắt đầu giơ chữ L
        start_t = time.time()
        with patch("time.time", return_value=start_t):
            self.controller.process(self.frame, [hand_l])
            self.assertIsNotNone(self.controller.l_sign_start_time)
            self.assertTrue(self.action_queue.empty())

        # Frame 2: Giữ được 0.5s (chưa đủ 1.0s) -> Chưa bắn lệnh
        with patch("time.time", return_value=start_t + 0.50):
            self.controller.process(self.frame, [hand_l])
            self.assertTrue(self.action_queue.empty())

        # Frame 3: Giữ đủ 1.05s -> Bắn lệnh F5!
        with patch("time.time", return_value=start_t + 1.05):
            self.controller.process(self.frame, [hand_l])
            self.assertFalse(self.action_queue.empty())
            action = self.action_queue.get_nowait()
            self.assertIn(action["action"], ("PRESENTATION_START", "PRESENTATION_F5"))

    def test_slide_switch_workflow(self):
        """
        Kiểm tra quy trình chuyển slide chuẩn:
        1. Giơ bàn tay mở (4-5 ngón duỗi) giữ trong 1.0s để nạp (ARMING -> ARMED).
        2. Khi đã SẴN SÀNG (ARMED), quơ tay sang trái -> SWIPE_LEFT (Next Slide).
        3. Kiểm tra Cooldown 0.8s chống nhảy đúp slide.
        """
        wrist_init = (640.0, 400.0)
        hand_open = make_realistic_hand_pts(wrist=wrist_init, scale=120.0, fingers_up=(1, 1, 1, 1, 1))

        # 1. Bắt đầu giơ tay nạp
        t0 = time.time()
        with patch("time.time", return_value=t0):
            self.controller.process(self.frame, [hand_open])
            self.assertEqual(self.controller.slide_state, "ARMING")
            self.assertTrue(self.action_queue.empty())

        # 2. Sau 0.6s -> Vẫn đang nạp ARMING, chưa được quơ
        with patch("time.time", return_value=t0 + 0.6):
            self.controller.process(self.frame, [hand_open])
            self.assertEqual(self.controller.slide_state, "ARMING")
            self.assertTrue(self.action_queue.empty())

        # 3. Sau 1.05s -> Nạp hoàn tất! Chuyển sang ARMED
        t_armed = t0 + 1.05
        with patch("time.time", return_value=t_armed):
            self.controller.process(self.frame, [hand_open])
            self.assertEqual(self.controller.slide_state, "ARMED")
            self.assertIsNotNone(self.controller.arm_pos)
            self.assertTrue(self.action_queue.empty())

        # 4. Người dùng quơ tay sang trái (di chuyển wrist sang trái 90px)
        hand_swipe_left = make_realistic_hand_pts(wrist=(550.0, 400.0), scale=120.0, fingers_up=(1, 1, 1, 1, 1))
        t_swipe = t_armed + 0.10
        with patch("time.time", return_value=t_swipe):
            self.controller.process(self.frame, [hand_swipe_left])
            # Bắn lệnh SWIPE_LEFT (Next Slide)
            self.assertFalse(self.action_queue.empty())
            action = self.action_queue.get_nowait()
            self.assertEqual(action["action"], "SWIPE_LEFT")
            # Tự động chuyển sang COOLDOWN
            self.assertEqual(self.controller.slide_state, "COOLDOWN")

        # 5. Trong vòng 0.4s tiếp theo (đang cooldown) -> Di chuyển tiếp không được bắn thêm lệnh
        with patch("time.time", return_value=t_swipe + 0.40):
            self.controller.process(self.frame, [hand_swipe_left])
            self.assertTrue(self.action_queue.empty())
            self.assertEqual(self.controller.slide_state, "COOLDOWN")

    def test_slide_switch_swipe_right(self):
        """Kiểm tra quơ tay sang phải (Swipe Right -> Prev Slide)."""
        wrist_init = (640.0, 400.0)
        hand_open = make_realistic_hand_pts(wrist=wrist_init, scale=120.0, fingers_up=(1, 1, 1, 1, 1))

        t0 = time.time()
        # Nạp 1.0s
        with patch("time.time", return_value=t0):
            self.controller.process(self.frame, [hand_open])
        with patch("time.time", return_value=t0 + 1.05):
            self.controller.process(self.frame, [hand_open])
            self.assertEqual(self.controller.slide_state, "ARMED")

        # Quơ sang phải (di chuyển wrist sang phải +90px)
        hand_swipe_right = make_realistic_hand_pts(wrist=(730.0, 400.0), scale=120.0, fingers_up=(1, 1, 1, 1, 1))
        with patch("time.time", return_value=t0 + 1.15):
            self.controller.process(self.frame, [hand_swipe_right])
            self.assertFalse(self.action_queue.empty())
            action = self.action_queue.get_nowait()
            self.assertEqual(action["action"], "SWIPE_RIGHT")

    def test_disarm_on_hand_drop_or_fist(self):
        """Kiểm tra hủy trạng thái khi nắm tay lại hoặc hạ tay khỏi camera."""
        hand_open = make_realistic_hand_pts(wrist=(640.0, 400.0), scale=120.0, fingers_up=(1, 1, 1, 1, 1))
        t0 = time.time()

        # Đang nạp
        with patch("time.time", return_value=t0):
            self.controller.process(self.frame, [hand_open])
            self.assertEqual(self.controller.slide_state, "ARMING")

        # Nắm tay lại (0 ngón duỗi)
        hand_fist = make_realistic_hand_pts(wrist=(640.0, 400.0), scale=120.0, fingers_up=(0, 0, 0, 0, 0))
        with patch("time.time", return_value=t0 + 0.3):
            self.controller.process(self.frame, [hand_fist])
            # Phải tự động hủy nạp về IDLE
            self.assertEqual(self.controller.slide_state, "IDLE")
            self.assertIsNone(self.controller.arm_start_time)

    @patch("pyautogui.press")
    def test_dispatcher_presentation_actions(self, mock_press):
        """Kiểm tra ActionDispatcher thực thi phím nóng F5, Right, Left chính xác."""
        q = queue.Queue()
        dispatcher = ActionDispatcher(action_queue=q)

        # 1. Test PRESENTATION_START -> pyautogui.press("f5")
        dispatcher._dispatch("PRESENTATION_START", {})
        mock_press.assert_called_with("f5")

        # 2. Test SWIPE_LEFT -> pyautogui.press("right")
        dispatcher._dispatch("SWIPE_LEFT", {})
        mock_press.assert_called_with("right")

        # 3. Test SWIPE_RIGHT -> pyautogui.press("left")
        dispatcher._dispatch("SWIPE_RIGHT", {})
        mock_press.assert_called_with("left")


if __name__ == "__main__":
    unittest.main()
