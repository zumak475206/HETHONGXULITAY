"""
tests/test_ui_settings.py
========================
Bộ kiểm thử tự động cho Tab 4: Cài Đặt Hệ Thống & Tinh Chỉnh Công Thái Học (SettingsPage).
Chạy hoàn toàn ở chế độ Headless (offscreen).
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Thiết lập chế độ offscreen cho Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from src.database.db_manager import DatabaseManager
from src.ui.tab_settings import SettingsPage, FACTORY_DEFAULTS
from src.ui.theme import MODERN_DARK_QSS


class TestUISettingsHeadless(unittest.TestCase):
    """Kiểm thử tự động giao diện SettingsPage, đồng bộ SQLite và điều khiển thời gian thực."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo QApplication duy nhất cho test suite."""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
            cls.app.setStyleSheet(MODERN_DARK_QSS)

    def setUp(self):
        """Tạo database SQLite tạm thời và mock popup dialogs."""
        self.patch_info = patch("PyQt6.QtWidgets.QMessageBox.information", return_value=None)
        self.patch_warn = patch("PyQt6.QtWidgets.QMessageBox.warning", return_value=None)
        self.patch_quest = patch("PyQt6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes)
        self.patch_info.start()
        self.patch_warn.start()
        self.patch_quest.start()

        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_settings.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.mock_camera_worker = MagicMock()
        self.settings_page = SettingsPage(
            db=self.db,
            author="AnhKy",
            camera_worker=self.mock_camera_worker,
        )
        self.settings_page.show()

    def tearDown(self):
        """Dọn dẹp widget và file tạm."""
        self.settings_page.close()
        self.settings_page.deleteLater()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.patch_info.stop()
        self.patch_warn.stop()
        self.patch_quest.stop()

    def test_settings_page_initialization(self):
        """Kiểm tra khởi tạo SettingsPage thành công với các controls đầy đủ."""
        self.assertIsNotNone(self.settings_page)
        self.assertEqual(self.settings_page.txt_author.text(), "AnhKy")
        self.assertEqual(self.settings_page.slider_sens.value(), int(FACTORY_DEFAULTS["mouse_sensitivity"] * 100))
        self.assertEqual(self.settings_page.slider_ygain.value(), int(FACTORY_DEFAULTS["mouse_y_gain"] * 100))
        self.assertEqual(self.settings_page.slider_beta.value(), int(FACTORY_DEFAULTS["one_euro_beta"] * 1000))
        self.assertEqual(self.settings_page.slider_min_cutoff.value(), int(FACTORY_DEFAULTS["one_euro_min_cutoff"] * 10))
        self.assertEqual(self.settings_page.slider_right_click.value(), int(FACTORY_DEFAULTS["right_click_hold_time"] * 100))
        self.assertEqual(self.settings_page.slider_fist_grace.value(), int(FACTORY_DEFAULTS["fist_grace_time"] * 100))
        self.assertEqual(self.settings_page.slider_switch_mode.value(), int(FACTORY_DEFAULTS["two_hand_switch_time"] * 10))
        self.assertEqual(self.settings_page.cbo_res.currentIndex(), 0)  # 1280x720
        self.assertTrue(self.settings_page.chk_clahe.isChecked())

    def test_slider_value_change_updates_labels(self):
        """Kiểm tra kéo các thanh trượt lập tức cập nhật nội dung nhãn hiển thị."""
        # 1. Slider Độ nhạy chuột
        self.settings_page.slider_sens.setValue(185)
        self.assertIn("1.85x", self.settings_page.lbl_sens_val.text())

        # 2. Slider Y-Gain
        self.settings_page.slider_ygain.setValue(210)
        self.assertIn("2.10x", self.settings_page.lbl_ygain_val.text())

        # 3. Slider Beta
        self.settings_page.slider_beta.setValue(65)
        self.assertIn("0.065", self.settings_page.lbl_beta_val.text())

        # 4. Slider Min Cutoff
        self.settings_page.slider_min_cutoff.setValue(18)
        self.assertIn("1.8 Hz", self.settings_page.lbl_cutoff_val.text())

        # 5. Slider Click Phải
        self.settings_page.slider_right_click.setValue(45)
        self.assertIn("0.45s", self.settings_page.lbl_rc_val.text())

        # 6. Slider Fist Grace
        self.settings_page.slider_fist_grace.setValue(35)
        self.assertIn("0.35s", self.settings_page.lbl_fist_val.text())

        # 7. Slider Switch Mode
        self.settings_page.slider_switch_mode.setValue(25)
        self.assertIn("2.5s", self.settings_page.lbl_switch_val.text())

    def test_save_settings_persists_to_sqlite(self):
        """Kiểm tra lưu cài đặt ghi chuẩn xác xuống bảng app_settings trong SQLite."""
        self.settings_page.txt_author.setText("NguyenVanTest")
        self.settings_page.slider_sens.setValue(200)       # 2.00x
        self.settings_page.slider_ygain.setValue(180)      # 1.80x
        self.settings_page.slider_beta.setValue(50)        # 0.050
        self.settings_page.slider_min_cutoff.setValue(15)  # 1.5 Hz
        self.settings_page.slider_right_click.setValue(40) # 0.40s
        self.settings_page.slider_fist_grace.setValue(30)  # 0.30s
        self.settings_page.slider_switch_mode.setValue(30) # 3.0s
        self.settings_page.cbo_res.setCurrentIndex(1)      # 640x480
        self.settings_page.chk_clahe.setChecked(False)

        self.settings_page.save_settings()

        # Kiểm tra đọc từ SQLite
        self.assertEqual(self.db.get_setting("active_author"), "NguyenVanTest")
        self.assertEqual(self.db.get_setting("mouse_sensitivity"), "2.00")
        self.assertEqual(self.db.get_setting("mouse_y_gain"), "1.80")
        self.assertEqual(self.db.get_setting("one_euro_beta"), "0.0500")
        self.assertEqual(self.db.get_setting("one_euro_min_cutoff"), "1.50")
        self.assertEqual(self.db.get_setting("right_click_hold_time"), "0.40")
        self.assertEqual(self.db.get_setting("fist_grace_time"), "0.30")
        self.assertEqual(self.db.get_setting("two_hand_switch_time"), "3.00")
        self.assertEqual(self.db.get_setting("camera_width"), "640")
        self.assertEqual(self.db.get_setting("camera_height"), "480")
        self.assertEqual(self.db.get_setting("clahe_enabled"), "0")

    def test_author_changed_and_settings_saved_signals(self):
        """Kiểm tra phát các PyQt Signals khi lưu cài đặt."""
        author_signal_received = []
        settings_signal_received = []

        self.settings_page.author_changed.connect(lambda a: author_signal_received.append(a))
        self.settings_page.settings_saved.connect(lambda s: settings_signal_received.append(s))

        self.settings_page.txt_author.setText("HeroAuthor")
        self.settings_page.slider_sens.setValue(150)
        self.settings_page.save_settings()

        self.assertEqual(len(author_signal_received), 1)
        self.assertEqual(author_signal_received[0], "HeroAuthor")
        self.assertEqual(len(settings_signal_received), 1)
        self.assertEqual(settings_signal_received[0]["mouse_sensitivity"], 1.50)
        self.assertEqual(settings_signal_received[0]["active_author"], "HeroAuthor")

    def test_live_dispatch_to_camera_worker(self):
        """Kiểm tra gọi cập nhật tham số tức thì tới CameraWorker."""
        self.mock_camera_worker.reset_mock()

        self.settings_page.slider_sens.setValue(160)
        self.settings_page.slider_ygain.setValue(170)
        self.settings_page.chk_clahe.setChecked(True)
        self.settings_page.slider_switch_mode.setValue(22)  # 2.2s

        self.settings_page.save_settings()

        # CameraWorker.update_mouse_settings phải được gọi
        self.mock_camera_worker.update_mouse_settings.assert_called_with(
            sensitivity=1.60,
            y_gain=1.70,
            beta=0.040,
            min_cutoff=1.0,
            right_click_hold_time=0.35,
            fist_grace_time=0.25,
        )
        self.mock_camera_worker.set_clahe_enabled.assert_called_with(True)
        self.mock_camera_worker.set_two_hand_switch_time.assert_called_with(2.2)

    def test_factory_reset_restores_defaults(self):
        """Kiểm tra tính năng Khôi phục mặc định (Factory Reset)."""
        # Thay đổi sang các giá trị khác lạ
        self.settings_page.slider_sens.setValue(240)
        self.settings_page.slider_ygain.setValue(230)
        self.settings_page.slider_beta.setValue(120)
        self.settings_page.slider_min_cutoff.setValue(25)
        self.settings_page.chk_clahe.setChecked(False)

        # Kích hoạt Factory Reset
        self.settings_page.reset_defaults()

        # Kiểm tra controls đã trở về Factory Defaults
        self.assertEqual(self.settings_page.slider_sens.value(), int(FACTORY_DEFAULTS["mouse_sensitivity"] * 100))
        self.assertEqual(self.settings_page.slider_ygain.value(), int(FACTORY_DEFAULTS["mouse_y_gain"] * 100))
        self.assertEqual(self.settings_page.slider_beta.value(), int(FACTORY_DEFAULTS["one_euro_beta"] * 1000))
        self.assertEqual(self.settings_page.slider_min_cutoff.value(), int(FACTORY_DEFAULTS["one_euro_min_cutoff"] * 10))
        self.assertTrue(self.settings_page.chk_clahe.isChecked())

        # Kiểm tra SQLite cũng đã cập nhật
        self.assertEqual(self.db.get_setting("mouse_sensitivity"), f"{FACTORY_DEFAULTS['mouse_sensitivity']:.2f}")
        self.assertEqual(self.db.get_setting("clahe_enabled"), "1")

    def test_set_camera_worker_after_init(self):
        """Kiểm tra gắn CameraWorker sau khi khởi tạo trang và kích hoạt dispatch ngay."""
        fresh_page = SettingsPage(db=self.db, author="LateWorker", camera_worker=None)
        late_worker = MagicMock()
        fresh_page.set_camera_worker(late_worker)

        late_worker.update_mouse_settings.assert_called_once()
        late_worker.set_clahe_enabled.assert_called_once()
        fresh_page.close()
        fresh_page.deleteLater()


if __name__ == "__main__":
    unittest.main()
