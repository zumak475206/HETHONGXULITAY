"""
tests/test_ui_recorder_dialog.py
================================
Bộ kiểm thử tự động cho Hộp thoại Thu Thập Mẫu Trực Quan (RecordingDialog).
Kiểm tra luồng Camera live preview, đếm ngược, nắn chuẩn thời gian 45 frames (Resampling),
tự động lưu SQLite và nút xóa mẫu vừa quay khi bị lỗi.
Vận hành hoàn toàn ở chế độ Headless (offscreen).
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Thiết lập chế độ offscreen cho Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from src.database.db_manager import DatabaseManager
from src.database.dataset_merger import resample_sequence
from src.ui.dialog_recorder import RecordingDialog
from src.ui.theme import MODERN_DARK_QSS


class TestUIRecorderDialogHeadless(unittest.TestCase):
    """Kiểm thử tự động giao diện RecordingDialog và thuật toán Temporal Resampling."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo QApplication duy nhất cho test suite."""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
            cls.app.setStyleSheet(MODERN_DARK_QSS)

    def setUp(self):
        """Tạo môi trường database cô lập và mock camera worker."""
        self.patch_info = patch("PyQt6.QtWidgets.QMessageBox.information", return_value=None)
        self.patch_warn = patch("PyQt6.QtWidgets.QMessageBox.warning", return_value=None)
        self.patch_info.start()
        self.patch_warn.start()

        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_recorder.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.mock_worker = MagicMock()
        self.gesture_info = {
            "name": "TEST_VSL_HELLO",
            "display_name": "Xin chào Test",
            "finger_guide": "Vẫy nhẹ 5 ngón tay",
            "mode_id": 3,
        }
        self.dialog = RecordingDialog(
            db=self.db,
            gesture_info=self.gesture_info,
            author="Tester",
            camera_worker=self.mock_worker,
        )
        self.dialog.show()

    def tearDown(self):
        """Dọn dẹp widget và file tạm."""
        self.dialog.close()
        self.dialog.deleteLater()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.patch_info.stop()
        self.patch_warn.stop()

    def test_dialog_initialization(self):
        """Kiểm tra khởi tạo hộp thoại đầy đủ các controls."""
        self.assertIsNotNone(self.dialog)
        self.assertEqual(self.dialog.gesture_name, "TEST_VSL_HELLO")
        self.assertEqual(self.dialog.mode_id, 3)
        self.assertEqual(self.dialog._target_duration_sec, 3.0)
        self.assertFalse(self.dialog.btn_stop.isEnabled())
        self.assertFalse(self.dialog.btn_delete_last.isEnabled())
        self.assertTrue(self.dialog.btn_record.isEnabled())

    def test_duration_combo_selection(self):
        """Kiểm tra thay đổi thời lượng trong ComboBox."""
        # 1. Nhanh: 1.5s
        self.dialog.cbo_duration.setCurrentIndex(1)
        self.assertEqual(self.dialog._target_duration_sec, 1.5)

        # 2. Dài: 5.0s
        self.dialog.cbo_duration.setCurrentIndex(2)
        self.assertEqual(self.dialog._target_duration_sec, 5.0)

        # 3. Tự do: 0.0s
        self.dialog.cbo_duration.setCurrentIndex(3)
        self.assertEqual(self.dialog._target_duration_sec, 0.0)

        # 4. Trở về Chuẩn: 3.0s
        self.dialog.cbo_duration.setCurrentIndex(0)
        self.assertEqual(self.dialog._target_duration_sec, 3.0)

    def test_camera_frame_rendering_all_states(self):
        """Kiểm tra vẽ khung hình camera trong các trạng thái IDLE, COUNTDOWN, RECORDING."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. IDLE
        self.dialog._state = "IDLE"
        self.dialog.on_camera_frame(dummy_frame)
        self.assertIsNotNone(self.dialog.lbl_camera.pixmap())

        # 2. COUNTDOWN
        self.dialog._state = "COUNTDOWN"
        self.dialog._countdown_val = 2
        self.dialog.on_camera_frame(dummy_frame)
        self.assertIsNotNone(self.dialog.lbl_camera.pixmap())

        # 3. RECORDING
        self.dialog._state = "RECORDING"
        self.dialog.on_camera_frame(dummy_frame)
        self.assertIsNotNone(self.dialog.lbl_camera.pixmap())

    def test_recording_cycle_and_temporal_resampling(self):
        """Kiểm tra chu trình quay 120 frames thô -> nắn chuẩn về 45 frames x 158D."""
        recorded_signals = []
        self.dialog.sample_recorded.connect(lambda g, p: recorded_signals.append((g, p)))

        # Bắt đầu quay
        self.dialog.start_recording()
        self.assertEqual(self.dialog._state, "COUNTDOWN")
        self.assertEqual(self.dialog._countdown_val, 3)

        # Đếm ngược 3 -> 2 -> 1 -> BẮT ĐẦU
        self.dialog._on_countdown_tick()  # 2
        self.dialog._on_countdown_tick()  # 1
        self.dialog._on_countdown_tick()  # 0 -> RECORDING
        self.assertEqual(self.dialog._state, "RECORDING")

        # Nạp 120 frames thô (giả lập thao tác dài 2-4 giây)
        for i in range(120):
            dummy_feat = [float(i)] * 158
            self.dialog.on_feature_frame(dummy_feat)

        self.assertEqual(len(self.dialog._record_buffer), 120)

        # Dừng và lưu
        self.dialog.stop_and_save()
        self.assertEqual(self.dialog._state, "IDLE")
        self.assertEqual(len(recorded_signals), 1)

        g_name, f_path = recorded_signals[0]
        self.assertEqual(g_name, "TEST_VSL_HELLO")
        self.assertTrue(os.path.exists(f_path))

        # Kiểm tra file đã lưu có đúng chuẩn (45, 158) nhờ Temporal Resampling
        data = np.load(f_path, allow_pickle=True)
        seq = data["sequence"]
        self.assertEqual(seq.shape, (45, 158))
        self.assertEqual(data["gesture_name"], "TEST_VSL_HELLO")
        self.assertEqual(data["author"], "Tester")

        # Kiểm tra nút Xóa Mẫu Vừa Quay đã được kích hoạt
        self.assertTrue(self.dialog.btn_delete_last.isEnabled())
        self.assertEqual(self.dialog._session_count, 1)

    def test_delete_last_sample_flow(self):
        """Kiểm tra nút Xóa Mẫu Vừa Quay khi thu thập bị lỗi."""
        deleted_signals = []
        self.dialog.sample_deleted.connect(lambda g, p: deleted_signals.append((g, p)))

        # Quay 1 mẫu hợp lệ
        self.dialog.start_recording()
        for _ in range(3):
            self.dialog._on_countdown_tick()

        for _ in range(50):
            self.dialog.on_feature_frame([0.5] * 158)

        self.dialog.stop_and_save()
        self.assertIsNotNone(self.dialog._last_saved_file)
        saved_file = Path(self.dialog._last_saved_file)
        self.assertTrue(saved_file.exists())

        # Bấm nút Xóa mẫu vừa quay
        self.dialog.delete_last_sample()

        # Kiểm tra file trên đĩa đã bị xóa
        self.assertFalse(saved_file.exists())
        self.assertIsNone(self.dialog._last_saved_file)
        self.assertFalse(self.dialog.btn_delete_last.isEnabled())
        self.assertEqual(self.dialog._session_count, 0)
        self.assertEqual(len(deleted_signals), 1)
        self.assertEqual(deleted_signals[0][0], "TEST_VSL_HELLO")

    def test_minimum_frame_guard(self):
        """Kiểm tra chặn lưu nếu số frames thu thập được quá ít (< 5 frames)."""
        self.dialog.start_recording()
        for _ in range(3):
            self.dialog._on_countdown_tick()

        # Chỉ nạp 2 frames
        self.dialog.on_feature_frame([0.1] * 158)
        self.dialog.on_feature_frame([0.2] * 158)

        self.dialog.stop_and_save()
        # Không được lưu
        self.assertIsNone(self.dialog._last_saved_file)
        self.assertEqual(self.dialog._session_count, 0)
        self.assertIn("Chưa đủ khung hình", self.dialog.lbl_status.text())

    def test_temporal_resampling_math(self):
        """Kiểm tra độ chính xác toán học của hàm resample_sequence."""
        # 1. Nén từ 100 frames xuống 45 frames
        t100 = np.linspace(0, 10, 100)[:, None]  # shape (100, 1)
        r45 = resample_sequence(t100, target_seq_len=45)
        self.assertEqual(r45.shape, (45, 1))
        self.assertAlmostEqual(float(r45[0, 0]), 0.0, places=3)
        self.assertAlmostEqual(float(r45[-1, 0]), 10.0, places=3)

        # 2. Giãn từ 20 frames lên 45 frames
        t20 = np.linspace(5, 15, 20)[:, None]   # shape (20, 1)
        r45_up = resample_sequence(t20, target_seq_len=45)
        self.assertEqual(r45_up.shape, (45, 1))
        self.assertAlmostEqual(float(r45_up[0, 0]), 5.0, places=3)
        self.assertAlmostEqual(float(r45_up[-1, 0]), 15.0, places=3)

    def test_mouse_override_on_dialog_lifecycle(self):
        """Kiểm tra hộp thoại tự động ngắt chuột khi mở và khôi phục khi đóng."""
        self.mock_worker.set_recording_override.assert_called_with(True, mode_id=3)
        self.dialog.close()
        self.mock_worker.set_recording_override.assert_called_with(False)

    def test_stop_button_during_countdown_cancels_cleanly(self):
        """Kiểm tra bấm nút Dừng/Hủy khi đang đếm ngược sẽ hủy nhẹ nhàng không báo lỗi."""
        self.dialog.start_recording()
        self.assertEqual(self.dialog._state, "COUNTDOWN")
        self.assertIn("Hủy Đếm Ngược", self.dialog.btn_stop.text())

        # Bấm Dừng khi đang đếm ngược
        self.dialog.btn_stop.click()
        self.assertEqual(self.dialog._state, "IDLE")
        self.assertFalse(self.dialog.btn_stop.isEnabled())
        self.assertTrue(self.dialog.btn_record.isEnabled())
        self.assertIn("Đã hủy đếm ngược", self.dialog.lbl_status.text())

    def test_spacebar_keyboard_shortcut(self):
        """Kiểm tra phím cách (Space) kích hoạt Bắt đầu và Dừng & Lưu ngay."""
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import QEvent

        # 1. Nhấn Space khi IDLE -> Bắt đầu COUNTDOWN
        press_space = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        self.dialog.keyPressEvent(press_space)
        self.assertEqual(self.dialog._state, "COUNTDOWN")

        # Đếm ngược tới RECORDING
        for _ in range(3):
            self.dialog._on_countdown_tick()
        self.assertEqual(self.dialog._state, "RECORDING")

        # Nạp dữ liệu
        for _ in range(15):
            self.dialog.on_feature_frame([0.7] * 158)

        # 2. Nhấn Space khi RECORDING -> Dừng và lưu ngay
        self.dialog.keyPressEvent(press_space)
        self.assertEqual(self.dialog._state, "IDLE")
        self.assertEqual(self.dialog._session_count, 1)
        self.assertIsNotNone(self.dialog._last_saved_file)


if __name__ == "__main__":
    unittest.main()
