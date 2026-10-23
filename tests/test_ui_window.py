"""
tests/test_ui_window.py
=======================
Bộ kiểm thử tự động cho Giao diện Desktop PyQt6 (MainWindow, Sidebar, Navigation, Mode Strip, HUD, Database Sync).
Sử dụng cờ offscreen để kiểm thử không cần màn hình đồ họa.
"""

import sys
import os
import unittest
from unittest.mock import patch
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Thiết lập chế độ offscreen cho Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.database.db_manager import DatabaseManager
from src.ui.main_window import MainWindow
from src.ui.theme import MODERN_DARK_QSS


class TestMainWindowHeadless(unittest.TestCase):
    """Kiểm thử tự động giao diện MainWindow ở chế độ Headless."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo QApplication duy nhất cho toàn bộ test suite."""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
            cls.app.setStyleSheet(MODERN_DARK_QSS)

    def setUp(self):
        """Tạo môi trường database cô lập cho mỗi test case và chặn modal popup."""
        self.patch_info = patch("src.ui.main_window.QMessageBox.information", return_value=None)
        self.patch_warn = patch("src.ui.main_window.QMessageBox.warning", return_value=None)
        self.patch_info.start()
        self.patch_warn.start()

        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_gestures.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.window = MainWindow(db=self.db, start_camera=False)

    def tearDown(self):
        """Dọn dẹp widget và file tạm."""
        self.window.close()
        self.window.deleteLater()
        if hasattr(self.db, "conn") and self.db.conn:
            self.db.conn.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.patch_info.stop()
        self.patch_warn.stop()

    def test_window_initialization(self):
        """Kiểm tra khởi tạo cửa sổ và các thuộc tính cơ bản."""
        self.assertIsNotNone(self.window)
        self.assertIn("HETHONGXULITAY", self.window.windowTitle())
        self.assertEqual(self.window.stacked_widget.count(), 4)
        self.assertEqual(self.window.stacked_widget.currentIndex(), 0)

    def test_sidebar_navigation_switching(self):
        """Kiểm tra chuyển đổi các tab trang qua Sidebar buttons."""
        self.assertEqual(len(self.window.nav_buttons), 4)

        # 1. Chuyển sang Trang 1: Kho Cử chỉ & Thống kê
        self.window.nav_buttons[1].click()
        self.assertEqual(self.window.stacked_widget.currentIndex(), 1)
        self.assertEqual(self.window.nav_buttons[1].property("active"), "true")
        self.assertEqual(self.window.nav_buttons[0].property("active"), "false")

        # 2. Chuyển sang Trang 2: Studio Training AI
        self.window.nav_buttons[2].click()
        self.assertEqual(self.window.stacked_widget.currentIndex(), 2)
        self.assertEqual(self.window.nav_buttons[2].property("active"), "true")

        # 3. Chuyển sang Trang 3: Cài đặt hệ thống
        self.window.nav_buttons[3].click()
        self.assertEqual(self.window.stacked_widget.currentIndex(), 3)
        self.assertEqual(self.window.nav_buttons[3].property("active"), "true")

        # 4. Quay lại Trang 0: Trang chủ Live
        self.window.nav_buttons[0].click()
        self.assertEqual(self.window.stacked_widget.currentIndex(), 0)
        self.assertEqual(self.window.nav_buttons[0].property("active"), "true")

    def test_mode_selector_strip(self):
        """Kiểm tra thanh chọn chế độ dưới Camera."""
        self.assertIn("MODE1_MOUSE", self.window.mode_buttons)
        self.assertIn("MODE2_GESTURES", self.window.mode_buttons)
        self.assertIn("MODE3_SIGN", self.window.mode_buttons)

        # Chuyển sang Mode 2 bằng click button
        self.window.mode_buttons["MODE2_GESTURES"].click()
        self.assertEqual(self.window.current_mode_name, "MODE2_GESTURES")
        self.assertEqual(self.db.get_setting("current_mode"), "MODE2_GESTURES")

        # Chuyển sang Mode 3 bằng click button
        self.window.mode_buttons["MODE3_SIGN"].click()
        self.assertEqual(self.window.current_mode_name, "MODE3_SIGN")
        self.assertEqual(self.db.get_setting("current_mode"), "MODE3_SIGN")

        # Kiểm tra chế độ Tạm khóa Standby
        self.window.btn_standby.click()
        self.assertTrue(self.window.is_paused)
        self.window.btn_standby.click()
        self.assertFalse(self.window.is_paused)

    def test_camera_frame_and_hud_update(self):
        """Kiểm tra cập nhật frame OpenCV và các chỉ số HUD thời gian thực."""
        # Tạo frame giả lập (640, 480, 3)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[100:200, 100:200] = [0, 255, 0]  # Một ô xanh lá

        # Cập nhật frame lên camera canvas
        self.window.update_camera_frame(dummy_frame)
        self.assertIsNotNone(self.window.lbl_cam_preview.pixmap())

        # Cập nhật các chỉ số HUD
        self.window.update_hud_metrics(
            fps=59.4,
            action="Click trai (Pinch Lock)",
            pipeline="Dual-Pipeline (Direct Raw)",
            guard="AN TOAN (0 loi)",
        )
        self.assertIn("59.4 FPS", self.window.lbl_hud_fps.text())
        self.assertIn("Click trai", self.window.lbl_hud_action.text())
        self.assertIn("Dual-Pipeline", self.window.lbl_hud_pipeline.text())
        self.assertIn("AN TOAN", self.window.lbl_hud_guard.text())

    def test_studio_add_gesture(self):
        """Kiểm tra chức năng thêm cử chỉ tùy chỉnh từ tab Studio."""
        # Chuyển sang tab Studio
        self.window._switch_page(2)

        # Điền form
        self.window.txt_gesture_name.setText("CUSTOM_HELLO")
        self.window.txt_display_name.setText("Chào mừng Hello")
        self.window.txt_guide.setText("Xòe bàn tay vẫy nhẹ")

        # Thêm cử chỉ
        self.window._add_custom_gesture()

        # Kiểm tra đã lưu vào DB
        g = self.db.get_gesture("CUSTOM_HELLO")
        self.assertIsNotNone(g)
        self.assertEqual(g["display_name"], "Chào mừng Hello")
        self.assertEqual(g["is_system"], 0)

    def test_settings_save(self):
        """Kiểm tra lưu cài đặt hệ thống từ giao diện xuống SQLite."""
        # Chuyển sang tab cài đặt
        self.window._switch_page(3)

        # Đổi tên người đóng góp và độ nhạy
        self.window.txt_author.setText("NguyenVanA")
        self.window.slider_sens.setValue(180)  # 1.80x
        self.window._save_settings()

        # Kiểm tra đã lưu vào DB
        self.assertEqual(self.db.get_setting("active_author"), "NguyenVanA")
        self.assertEqual(self.db.get_setting("mouse_sensitivity"), "1.80")
        self.assertEqual(self.window.active_author, "NguyenVanA")


if __name__ == "__main__":
    unittest.main()
