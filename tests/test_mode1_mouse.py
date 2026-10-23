"""
tests/test_mode1_mouse.py
=========================
Unit tests kiểm thử toàn diện Mode 1 (Điều khiển chuột siêu mượt):
- Pinch Position Lock
- Drag & Drop State Machine
- Click phải Chữ V (V-Sign Hold 0.35s)
- Cần gạt Cuộn trang Anchor Joystick Scroll
- Fist Lock (Nắm đấm nghỉ tay) & Grace Period
- AsyncSystemController (Điều khiển âm lượng / độ sáng ngầm)
"""

import unittest
from unittest.mock import patch, MagicMock
import time
import math
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.modes.mode1_mouse import MouseController, AsyncSystemController
from tests.test_anatomical_guard import create_realistic_hand


class TestMode1Mouse(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.controller = MouseController(frame_w=640, frame_h=480)
        self.controller.screen_w = 1920
        self.controller.screen_h = 1080

    def tearDown(self):
        self.controller.cancel_click_and_drag()
        self.controller.sys_ctrl.stop()

    @patch("pyautogui.moveTo")
    def test_cursor_movement_and_smoothing(self, mock_move):
        """Kiểm tra di chuyển chuột cơ bản qua One Euro Filter."""
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Frame 1: Khởi tạo tọa độ
        self.controller.process(self.frame, hand)
        # Frame 2: Di chuyển tay sang phải
        hand2 = create_realistic_hand(wrist=(360.0, 300.0), scale=100.0)
        self.controller.process(self.frame, hand2)
        self.assertTrue(mock_move.called)

    @patch("pyautogui.click")
    @patch("pyautogui.moveTo")
    def test_pinch_position_lock_and_left_click(self, mock_move, mock_click):
        """
        Kiểm tra:
        1. Khi chạm ngón cái vào má ngón trỏ: Tọa độ con trỏ bị khóa cứng (Pinch Lock).
        2. Nhấp nhả nhanh (< 0.5s): Kích hoạt Click chuột trái tức thì.
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Chạm đầu ngón cái (4) vào khớp má ngón trỏ (6 - PIP)
        pip_pos = hand[6]
        hand[4] = (4, pip_pos[1] + 2.0, pip_pos[2] + 2.0, 0.0)

        # Frame 1: Chạm ngón cái -> Bắt đầu sạc click / pinch lock
        self.controller.process(self.frame, hand)
        self.assertIsNotNone(self.controller.locked_x)
        self.assertIsNotNone(self.controller.locked_y)
        locked_x = self.controller.locked_x
        locked_y = self.controller.locked_y

        # Frame 2: Tay hơi rung lắc nhẹ -> Con trỏ vẫn phải bị khóa nguyên vị trí
        hand_jitter = [list(pt) for pt in hand]
        hand_jitter[8][1] += 5.0  # Đầu ngón trỏ rung
        self.controller.process(self.frame, hand_jitter)
        self.assertEqual(self.controller.locked_x, locked_x)
        self.assertEqual(self.controller.locked_y, locked_y)

        # Frame 3: Nhấc ngón cái ra (< 0.5s) -> Ăn Click trái tức thì
        hand_release = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        self.controller.process(self.frame, hand_release)
        self.assertTrue(mock_click.called)
        self.assertIsNone(self.controller.locked_x)

    @patch("pyautogui.mouseDown")
    @patch("pyautogui.mouseUp")
    @patch("pyautogui.moveTo")
    def test_drag_and_drop_state_machine(self, mock_move, mock_up, mock_down):
        """
        Kiểm tra máy trạng thái Kéo-Thả (Drag & Drop):
        - Giữ ngón cái chạm ngón trỏ >= 0.5s -> kích hoạt mouseDown (Dragging).
        - Khi đang kéo thả: Mở khóa vị trí để di chuyển đối tượng tự do.
        - Nhấc ngón cái ra (> DROP_RELEASE 0.38) -> kích hoạt mouseUp.
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        pip_pos = hand[6]
        hand[4] = (4, pip_pos[1] + 2.0, pip_pos[2] + 2.0, 0.0)

        # Bắt đầu chạm
        self.controller.process(self.frame, hand)
        self.assertFalse(self.controller.is_dragging)

        # Giả lập thời gian trôi qua 0.55s
        self.controller.down_start_time = time.time() - 0.55
        self.controller.process(self.frame, hand)

        self.assertTrue(self.controller.is_dragging)
        self.assertTrue(mock_down.called)
        self.assertIsNone(self.controller.locked_x)  # Đã mở khóa để kéo

        # Nhấc ngón cái ra xa
        hand_released = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        self.controller.process(self.frame, hand_released)
        self.assertFalse(self.controller.is_dragging)
        self.assertTrue(mock_up.called)

    @patch("pyautogui.rightClick")
    @patch("pyautogui.moveTo")
    def test_v_sign_right_click(self, mock_move, mock_right_click):
        """
        Kiểm tra Click phải (Chữ V: 2 ngón Trỏ + Giữa xòe rộng):
        - Giữ đủ 0.35s -> Kích hoạt rightClick đúng 1 lần (có debounce và cooldown).
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Gập áp út (16) và út (20) vào lòng bàn tay để tạo tư thế 2 ngón (Trỏ + Giữa)
        mcp_wrist = hand[0]
        hand[16] = (16, hand[13][1], hand[13][2] + 20.0, 0.0)
        hand[20] = (20, hand[17][1], hand[17][2] + 20.0, 0.0)

        # Xòe rộng ngón giữa xa ngón trỏ (im_ratio >= 0.35)
        hand[12] = (12, hand[8][1] + 55.0, hand[8][2], 0.0)

        # Frame 1: Bắt đầu sạc chữ V
        self.controller.process(self.frame, hand)
        self.assertIsNotNone(self.controller.v_sign_hold_start)
        self.assertFalse(mock_right_click.called)

        # Giả lập thời gian giữ đủ 0.40s (> 0.35s)
        self.controller.v_sign_hold_start = time.time() - 0.40
        self.controller.last_right_click_time = time.time() - 2.0  # Hết cooldown
        self.controller.process(self.frame, hand)

        self.assertTrue(mock_right_click.called)
        self.assertTrue(self.controller.v_sign_clicked)

    @patch("pyautogui.scroll")
    @patch("pyautogui.moveTo")
    def test_anchor_joystick_scroll(self, mock_move, mock_scroll):
        """
        Kiểm tra Cần gạt Cuộn trang (Anchor Joystick Scroll):
        - Khép sát 2 ngón (Trỏ + Giữa) tạo tâm neo cố định.
        - Kéo lệch xuống ngoài deadzone -> Cuộn trang xuống (pyautogui.scroll âm).
        - Kéo lệch lên -> Cuộn trang lên (pyautogui.scroll dương).
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Gập áp út và út
        hand[16] = (16, hand[13][1], hand[13][2] + 20.0, 0.0)
        hand[20] = (20, hand[17][1], hand[17][2] + 20.0, 0.0)
        # Khép sát 2 ngón Trỏ và Giữa (im_ratio < 0.28)
        hand[12] = (12, hand[8][1] + 8.0, hand[8][2], 0.0)

        # Frame 1: Thiết lập tâm neo (Anchor)
        self.controller.process(self.frame, hand)
        self.assertIsNotNone(self.controller.scroll_anchor_x)
        self.assertIsNotNone(self.controller.scroll_anchor_y)

        # Frame 2: Kéo tay xuống dưới 30px (vượt deadzone 4%)
        hand_down = [list(pt) for pt in hand]
        hand_down[8][2] += 30.0
        hand_down[12][2] += 30.0

        # Kích hoạt tích lũy cuộn
        self.controller.last_scroll_time = time.time() - 0.1
        self.controller.process(self.frame, hand_down)
        self.assertTrue(mock_scroll.called)

    @patch("pyautogui.mouseUp")
    @patch("pyautogui.moveTo")
    def test_fist_lock_and_grace_period(self, mock_move, mock_up):
        """
        Kiểm tra Nắm đấm nghỉ tay (Fist Lock):
        - Khi nắm đấm: Nhả chuột, dừng di chuyển, đặt cờ grace buffer.
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Gập tất cả 4 ngón [Trỏ, Giữa, Áp út, Út] về phía cổ tay
        wx, wy = hand[0][1], hand[0][2]
        for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            mcp_x, mcp_y = hand[mcp_id][1], hand[mcp_id][2]
            hand[tip_id] = (tip_id, mcp_x, mcp_y + 10.0, 0.0)

        # Giả lập đang kéo dở
        self.controller.is_dragging = True
        self.controller.is_mouse_down = True

        self.controller.process(self.frame, hand)
        self.assertFalse(self.controller.is_dragging)
        self.assertTrue(self.controller.just_unfisted)
        self.assertEqual(self.controller.action_text, "NAM TAY (Khoa chuot)")

    @patch("pyautogui.click")
    @patch("pyautogui.mouseDown")
    @patch("pyautogui.moveTo")
    def test_quick_tap_click_without_false_drag(self, mock_move, mock_down, mock_click):
        """
        Kiểm tra: Khi nhấp chạm nhanh (< 0.15s) nhưng tay bị xê dịch nhẹ trong khung hình,
        hệ thống phải nhận diện đúng là Click trái thay vì hiểu nhầm thành Kéo thả (mouseDown).
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        pip_pos = hand[6]
        hand[4] = (4, pip_pos[1] + 1.0, pip_pos[2] + 1.0, 0.0)

        # Frame 1: Chạm ngón cái
        self.controller.process(self.frame, hand)
        self.assertFalse(self.controller.is_dragging)

        # Frame 2: Xê dịch tay nhẹ 6px trong camera (~60px màn hình), thời gian mới chỉ 0.08s
        hand_moved = [list(pt) for pt in hand]
        hand_moved[8][1] += 6.0
        self.controller.process(self.frame, hand_moved)
        self.assertFalse(self.controller.is_dragging)
        self.assertFalse(mock_down.called)

        # Frame 3: Nhấc ngón cái ra -> Ăn click trái, KHÔNG BAO GIỜ kích hoạt drag
        hand_released = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        self.controller.process(self.frame, hand_released)
        self.assertTrue(mock_click.called)
        self.assertFalse(self.controller.is_dragging)

    @patch("pyautogui.moveTo")
    def test_v_sign_exit_unfreezes_cursor(self, mock_move):
        """
        Kiểm tra: Khi thoát khỏi tư thế chữ V, con trỏ chuột không bị kẹt (locked_x/locked_y được reset về None).
        """
        hand = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        # Gập áp út và út
        hand[16] = (16, hand[13][1], hand[13][2] + 20.0, 0.0)
        hand[20] = (20, hand[17][1], hand[17][2] + 20.0, 0.0)
        # Xòe rộng Trỏ và Giữa (V-sign)
        hand[12] = (12, hand[8][1] + 55.0, hand[8][2], 0.0)

        # Vào tư thế V-sign: con trỏ bị khóa vị trí để click chuẩn xác
        self.controller.process(self.frame, hand)
        self.assertIsNotNone(self.controller.locked_x)

        # Thoát khỏi tư thế V-sign trở về bàn tay xòe tự do (sau 3-4 frames làm mượt)
        hand_open = create_realistic_hand(wrist=(320.0, 300.0), scale=100.0)
        for _ in range(4):
            self.controller.process(self.frame, hand_open)

        # Khóa phải được gỡ bỏ ngay lập tức
        self.assertIsNone(self.controller.locked_x)
        self.assertIsNone(self.controller.locked_y)

    def test_async_system_controller(self):
        """Kiểm tra AsyncSystemController điều khiển âm lượng và độ sáng ngầm."""
        ctrl = AsyncSystemController()
        ctrl.set_volume(0.85)
        ctrl.set_brightness(70)

        self.assertEqual(ctrl.get_vol_pct(), 85)
        self.assertEqual(ctrl.get_bright_pct(), 70)
        ctrl.stop()


if __name__ == "__main__":
    unittest.main()
