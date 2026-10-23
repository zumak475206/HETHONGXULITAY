"""
tests/test_ui_studio.py
=======================
Bộ kiểm thử tự động cho Tab 3: Studio Thu Thập, Thêm Ký Hiệu Mới & Huấn Luyện AI (StudioPage).
Vận hành ở chế độ Headless (offscreen).
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

from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox
from PyQt6.QtCore import Qt

from src.database.db_manager import DatabaseManager
from src.ui.tab_studio import StudioPage
from src.ui.theme import MODERN_DARK_QSS


class TestUIStudioHeadless(unittest.TestCase):
    """Kiểm thử tự động giao diện StudioPage và luồng thu thập/huấn luyện."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo QApplication duy nhất cho test suite."""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
            cls.app.setStyleSheet(MODERN_DARK_QSS)

    def setUp(self):
        """Tạo database SQLite tạm thời và cô lập cho mỗi test case."""
        self.patch_info = patch("PyQt6.QtWidgets.QMessageBox.information", return_value=None)
        self.patch_warn = patch("PyQt6.QtWidgets.QMessageBox.warning", return_value=None)
        self.patch_info.start()
        self.patch_warn.start()

        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_studio.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.studio_page = StudioPage(db=self.db, author="TestUser")
        self.studio_page.show()

    def tearDown(self):
        """Dọn dẹp widget và file tạm."""
        if hasattr(self.studio_page, "_countdown_timer") and self.studio_page._countdown_timer.isActive():
            self.studio_page._countdown_timer.stop()
        if hasattr(self.studio_page, "train_worker") and self.studio_page.train_worker:
            self.studio_page.train_worker.terminate()
            self.studio_page.train_worker.wait(500)
        self.studio_page.close()
        self.studio_page.deleteLater()
        if hasattr(self.db, "conn") and self.db.conn:
            self.db.conn.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.patch_info.stop()
        self.patch_warn.stop()

    def test_studio_page_creation(self):
        """Kiểm tra khởi tạo StudioPage thành công và có đủ các thành phần 4 phân hệ."""
        self.assertIsNotNone(self.studio_page)
        # 1. Thêm ký hiệu
        self.assertIsNotNone(self.studio_page.txt_gesture_name)
        self.assertIsNotNone(self.studio_page.txt_display_name)
        self.assertIsNotNone(self.studio_page.cbo_mode)
        # 2. Thu thập mẫu
        self.assertIsNotNone(self.studio_page.cbo_record_gesture)
        self.assertIsNotNone(self.studio_page.progress_record)
        self.assertIsNotNone(self.studio_page.btn_start_record)
        self.assertIsNotNone(self.studio_page.btn_cancel_record)
        # 3. Huấn luyện AI
        self.assertIsNotNone(self.studio_page.txt_epochs)
        self.assertIsNotNone(self.studio_page.txt_batch)
        self.assertIsNotNone(self.studio_page.btn_train_ai)
        self.assertIsNotNone(self.studio_page.progress_train)
        self.assertIsNotNone(self.studio_page.lbl_train_status)

    def test_add_custom_gesture(self):
        """Kiểm tra thêm cử chỉ mới: validation, lưu SQLite, phát tín hiệu và refresh dropdown."""
        # 1. Kiểm tra validation khi form rỗng
        self.studio_page.txt_gesture_name.setText("")
        self.studio_page.txt_display_name.setText("")
        self.studio_page.add_custom_gesture()
        self.assertIsNone(self.db.get_gesture("CUSTOM_TEST"))

        # 2. Điền form hợp lệ
        self.studio_page.txt_gesture_name.setText("CUSTOM_TEST")
        self.studio_page.txt_display_name.setText("Cử chỉ Test Thử Nghiệm")
        self.studio_page.txt_guide.setText("Xòe 3 ngón tay")
        self.studio_page.cbo_mode.setCurrentIndex(2)  # Mode 3

        added_signals = []
        self.studio_page.gesture_added.connect(lambda name: added_signals.append(name))

        self.studio_page.add_custom_gesture()

        # Kiểm tra đã lưu vào DB
        g = self.db.get_gesture("CUSTOM_TEST")
        self.assertIsNotNone(g)
        self.assertEqual(g["display_name"], "Cử chỉ Test Thử Nghiệm")
        self.assertEqual(g["mode_id"], 3)
        self.assertEqual(g["is_system"], 0)
        self.assertEqual(added_signals, ["CUSTOM_TEST"])

        # Kiểm tra dropdown cử chỉ đã có mục mới
        items = [self.studio_page.cbo_record_gesture.itemText(i) for i in range(self.studio_page.cbo_record_gesture.count())]
        self.assertTrue(any("CUSTOM_TEST" in it for it in items))

    def test_record_sample_countdown_and_collection(self):
        """Kiểm tra chu trình quay mẫu: Đếm ngược 3-2-1 -> Gom 45 frames 158D -> Lưu .npz."""
        # Chọn một cử chỉ
        self.studio_page.cbo_record_gesture.setCurrentIndex(0)
        g_name = self.studio_page.get_selected_record_gesture()
        self.assertTrue(len(g_name) > 0)

        # Bắt đầu quay
        self.studio_page.start_recording()
        self.assertTrue(self.studio_page._countdown_timer.isActive())
        self.assertEqual(self.studio_page._countdown_val, 3)
        self.assertFalse(self.studio_page._is_recording)

        # Mô phỏng 3 nhịp đếm ngược
        self.studio_page._on_countdown_tick()  # -> 2
        self.assertEqual(self.studio_page._countdown_val, 2)
        self.studio_page._on_countdown_tick()  # -> 1
        self.assertEqual(self.studio_page._countdown_val, 1)
        self.studio_page._on_countdown_tick()  # -> 0: BẮT ĐẦU RECORD
        self.assertTrue(self.studio_page._is_recording)

        saved_signals = []
        self.studio_page.sample_recorded.connect(lambda name: saved_signals.append(name))

        # Bơm 45 frames 158D
        dummy_feat = [0.1] * 158
        for i in range(45):
            self.studio_page.on_feature_frame(dummy_feat)

        # Sau khi đủ 45 frames: đã hoàn thành và lưu mẫu
        self.assertFalse(self.studio_page._is_recording)
        self.assertEqual(saved_signals, [g_name])
        self.assertIn("Đã lưu mẫu thành công", self.studio_page.lbl_record_status.text())

        # Kiểm tra mẫu đã được ghi vào SQLite
        samples = self.db.get_samples_for_gesture(g_name)
        self.assertTrue(len(samples) >= 1)
        self.assertEqual(samples[-1]["author"], "TestUser")
        self.assertEqual(samples[-1]["num_frames"], 45)

    def test_record_sample_cancel(self):
        """Kiểm tra hủy chu trình quay mẫu."""
        self.studio_page.start_recording()
        self.studio_page.cancel_recording()
        self.assertFalse(self.studio_page._is_recording)
        self.assertFalse(self.studio_page._countdown_timer.isActive())
        self.assertTrue(self.studio_page.btn_start_record.isEnabled())
        self.assertFalse(self.studio_page.btn_cancel_record.isEnabled())

    def test_feed_sample_sequence_direct(self):
        """Kiểm tra nạp trực tiếp chuỗi mẫu 45x158 (dành cho headless/import)."""
        self.studio_page.cbo_record_gesture.setCurrentIndex(0)
        g_name = self.studio_page.get_selected_record_gesture()

        dummy_seq = np.random.randn(45, 158).astype(np.float32)
        out_path = self.studio_page.feed_sample_sequence(dummy_seq)

        self.assertTrue(out_path.exists())
        loaded = np.load(out_path)
        self.assertEqual(loaded["sequence"].shape, (45, 158))
        self.assertEqual(str(loaded["gesture_name"]), g_name)

    def test_training_progress_and_completion(self):
        """Kiểm tra phản hồi tiến trình và hoàn tất huấn luyện AI."""
        # Gọi trực tiếp các callback tiến trình
        self.studio_page._on_train_epoch_progress(ep=10, total_epochs=20, loss=0.25, acc=88.5)
        self.assertEqual(self.studio_page.progress_train.value(), 50)
        self.assertIn("Epoch 10/20", self.studio_page.lbl_train_status.text())

        # Giả lập hoàn thành
        dummy_summary = {
            "total_samples": 120,
            "total_classes": 10,
            "final_loss": 0.08,
            "final_acc": 96.5,
            "model_path": "gesture_model.onnx",
        }
        finished_signals = []
        self.studio_page.training_finished.connect(lambda s: finished_signals.append(s))

        self.studio_page._on_train_completed(dummy_summary)
        self.assertEqual(self.studio_page.progress_train.value(), 100)
        self.assertIn("Hoàn tất huấn luyện", self.studio_page.lbl_train_status.text())
        self.assertEqual(len(finished_signals), 1)

    def test_export_and_import_zip(self):
        """Kiểm tra tính năng đóng gói ZIP và nhập gói bạn bè."""
        # Tạo 1 mẫu giả lập
        dummy_seq = np.zeros((45, 158), dtype=np.float32)
        self.studio_page.feed_sample_sequence(dummy_seq)

        # Xuất zip giả lập
        zip_path = Path(self.test_dir) / "ExportTest.zip"
        with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(zip_path), "ZIP Files (*.zip)")):
            self.studio_page.export_my_zip()

        # Nhập zip giả lập
        with patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(zip_path), "ZIP Files (*.zip)")):
            self.studio_page.import_friend_zip()


if __name__ == "__main__":
    unittest.main()
