"""
tests/test_ui_workers.py
========================
Bộ kiểm thử tự động cho các luồng nền PyQt6 (CameraWorker, TrainWorker) và Cửa sổ Mini PiP (PiPWidget).
Kiểm thử tín hiệu (Signals), chuyển đổi chế độ, tiến trình huấn luyện và dọn dẹp tài nguyên.
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

import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer

from src.database.db_manager import DatabaseManager
from src.ui.pip_widget import PiPWidget
from src.ui.workers import CameraWorker, TrainWorker
from src.ui.main_window import MainWindow


class TestPiPWidget(unittest.TestCase):
    """Kiểm thử Cửa sổ nổi Mini Floating Camera (PiPWidget)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.pip = PiPWidget()

    def tearDown(self):
        self.pip.close()
        self.pip.deleteLater()

    def test_pip_initialization(self):
        """Kiểm tra khởi tạo và cờ cửa sổ PiP."""
        self.assertTrue(self.pip.windowFlags() & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(self.pip.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        self.assertEqual(self.pip.width(), 360)
        self.assertEqual(self.pip.height(), 230)

    def test_pip_badge_and_frame_update(self):
        """Kiểm tra cập nhật huy hiệu và khung hình camera trên PiP."""
        self.pip.set_mode_badge("MODE 2: TRÌNH CHIẾU", "#06b6d4")
        self.assertIn("TRÌNH CHIẾU", self.pip.lbl_badge.text())

        # Tạo QPixmap giả lập
        img = QImage(320, 180, QImage.Format.Format_RGB888)
        img.fill(Qt.GlobalColor.darkBlue)
        pix = QPixmap.fromImage(img)
        self.pip.update_frame(pix)
        self.assertIsNotNone(self.pip.lbl_cam.pixmap())

    def test_pip_restore_signal(self):
        """Kiểm tra tín hiệu phóng to khôi phục cửa sổ chính."""
        restored = []
        self.pip.restore_requested.connect(lambda: restored.append(True))
        self.pip.btn_restore.click()
        self.assertEqual(len(restored), 1)


class TestTrainWorker(unittest.TestCase):
    """Kiểm thử luồng nền huấn luyện AI (TrainWorker)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_gestures.db"
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        if hasattr(self.db, "conn") and self.db.conn:
            self.db.conn.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_training_worker_lifecycle(self):
        """Kiểm tra tiến trình chạy và phát tín hiệu hoàn tất của TrainWorker."""
        worker = TrainWorker(db=self.db, epochs=3, batch_size=16)

        epochs_received = []
        completed_results = []

        worker.epoch_progress.connect(lambda ep, tot, loss, acc: epochs_received.append((ep, tot)))
        worker.training_completed.connect(lambda res: completed_results.append(res))

        # Chạy trực tiếp run() để kiểm thử đồng bộ
        worker.run()

        self.assertEqual(len(epochs_received), 3)
        self.assertEqual(len(completed_results), 1)
        self.assertIn("final_acc", completed_results[0])
        self.assertGreater(completed_results[0]["final_acc"], 50.0)


class TestCameraWorkerControls(unittest.TestCase):
    """Kiểm thử các phương thức điều khiển trạng thái của CameraWorker."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.worker = CameraWorker(initial_mode="MODE1_MOUSE")

    def tearDown(self):
        self.worker.stop()
        self.worker.deleteLater()

    def test_set_mode_and_paused(self):
        """Kiểm tra đổi chế độ và tạm dừng trên CameraWorker."""
        self.assertEqual(self.worker.current_mode, "MODE1_MOUSE")

        self.worker.set_mode("MODE2_GESTURES")
        self.assertEqual(self.worker.current_mode, "MODE2_GESTURES")

        self.worker.set_paused(True)
        self.assertTrue(self.worker.is_paused)

        self.worker.set_paused(False)
        self.assertFalse(self.worker.is_paused)

    def test_set_mouse_paused_and_recording_override(self):
        """Kiểm tra tạm dừng riêng cho chuột và override mode_id khi thu thập mẫu."""
        self.assertFalse(self.worker.mouse_paused)
        self.assertIsNone(self.worker.recording_mode_id)

        # 1. Tạm dừng chuột riêng biệt
        self.worker.set_mouse_paused(True)
        self.assertTrue(self.worker.mouse_paused)

        self.worker.set_mouse_paused(False)
        self.assertFalse(self.worker.mouse_paused)

        # 2. Recording override
        self.worker.set_recording_override(True, mode_id=3)
        self.assertTrue(self.worker.mouse_paused)
        self.assertEqual(self.worker.recording_mode_id, 3)

        self.worker.set_recording_override(False)
        self.assertFalse(self.worker.mouse_paused)
        self.assertIsNone(self.worker.recording_mode_id)


class TestMainWindowPiPAndIntegration(unittest.TestCase):
    """Kiểm thử tích hợp MainWindow với PiP và các sự kiện luồng."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.patch_info = patch("src.ui.main_window.QMessageBox.information", return_value=None)
        self.patch_warn = patch("src.ui.main_window.QMessageBox.warning", return_value=None)
        self.patch_info.start()
        self.patch_warn.start()

        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_gestures.db"
        self.db = DatabaseManager(db_path=self.db_path)
        # Khởi tạo MainWindow với start_camera=False để kiểm tra UI an toàn
        self.window = MainWindow(db=self.db, start_camera=False)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        if hasattr(self.db, "conn") and self.db.conn:
            self.db.conn.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.patch_info.stop()
        self.patch_warn.stop()

    def test_toggle_pip_lifecycle(self):
        """Kiểm tra bật / tắt cửa sổ PiP từ nút bấm trên Trang chủ."""
        self.assertFalse(self.window.pip_widget.isVisible())

        # Bật PiP
        self.window.btn_pip.click()
        self.assertTrue(self.window.pip_widget.isVisible())
        self.assertIn("Đang Mở Mini", self.window.btn_pip.text())

        # Tắt PiP qua restore
        self.window.pip_widget._on_restore()
        self.assertFalse(self.window.pip_widget.isVisible())
        self.assertIn("Thu Nhỏ Mini", self.window.btn_pip.text())

    def test_worker_signals_to_ui(self):
        """Kiểm tra khi Worker phát tín hiệu frame, metrics, mode thì UI cập nhật đúng."""
        # 1. Tín hiệu frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.window._on_worker_frame_ready(dummy_frame)
        self.assertIsNotNone(self.window.lbl_cam_preview.pixmap())

        # 2. Tín hiệu metrics
        self.window._on_worker_metrics_ready(60.2, 8.5, "Di chuot (0ms)", "Zero-Lag Raw", "An toàn")
        self.assertIn("60.2 FPS", self.window.lbl_hud_fps.text())
        self.assertIn("Di chuot", self.window.lbl_hud_action.text())
        self.assertIn("Zero-Lag", self.window.lbl_hud_pipeline.text())

        # 3. Tín hiệu chuyển mode bằng cử chỉ (2 tay mở)
        self.window._on_worker_mode_switched("MODE3_SIGN")
        self.assertEqual(self.window.current_mode_name, "MODE3_SIGN")
        self.assertEqual(self.db.get_setting("current_mode"), "MODE3_SIGN")
        self.assertEqual(self.window.mode_buttons["MODE3_SIGN"].property("active"), "true")

    def test_switch_page_controls_mouse_pause(self):
        """Kiểm tra khi chuyển tab thì tự động tạm dừng chuột để tương tác UI."""
        # Giả lập camera_worker gắn vào main_window
        mock_worker = MagicMock()
        self.window.camera_worker = mock_worker

        # 1. Chuyển sang Tab 2 (Studio)
        self.window._switch_page(2)
        mock_worker.set_mouse_paused.assert_called_with(True)

        # 2. Chuyển sang Tab 1 (Library)
        self.window._switch_page(1)
        mock_worker.set_mouse_paused.assert_called_with(True)

        # 3. Chuyển về Tab 0 (Home)
        self.window.is_paused = False
        self.window._switch_page(0)
        mock_worker.set_mouse_paused.assert_called_with(False)


if __name__ == "__main__":
    unittest.main()
