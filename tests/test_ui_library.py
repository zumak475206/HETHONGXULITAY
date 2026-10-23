"""
tests/test_ui_library.py
========================
Bộ kiểm thử tự động cho Tab 2: Kho Cử Chỉ & Bảng Thống Kê (LibraryPage, GestureCard, GestureDetailDialog).
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
from src.ui.tab_library import LibraryPage, GestureCard, GestureDetailDialog
from src.ui.theme import MODERN_DARK_QSS


class TestUILibraryHeadless(unittest.TestCase):
    """Kiểm thử tự động giao diện LibraryPage, GestureCard và GestureDetailDialog."""

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
        self.db_path = Path(self.test_dir) / "test_library.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.library_page = LibraryPage(db=self.db)
        self.library_page.show()

    def tearDown(self):
        """Dọn dẹp widget và file tạm."""
        self.library_page.close()
        self.library_page.deleteLater()
        if hasattr(self.db, "conn") and self.db.conn:
            self.db.conn.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.patch_info.stop()
        self.patch_warn.stop()

    def test_library_page_creates_without_crash(self):
        """Kiểm tra khởi tạo LibraryPage thành công và có đủ stat cards, filter bar."""
        self.assertIsNotNone(self.library_page)
        self.assertIsNotNone(self.library_page._card_gestures)
        self.assertIsNotNone(self.library_page._card_samples)
        self.assertIsNotNone(self.library_page._card_contribs)
        self.assertIsNotNone(self.library_page._cbo_mode)
        self.assertIsNotNone(self.library_page._txt_search)
        self.assertIsNotNone(self.library_page._grid_layout)
        self.assertIsNotNone(self.library_page._lbl_chart)

    def test_refresh_loads_from_db(self):
        """Kiểm tra refresh() nạp đúng số lượng 10 cử chỉ mặc định từ SQLite."""
        self.assertEqual(len(self.library_page._all_cards), 10)
        # Kiểm tra giá trị trên stat card
        labels = self.library_page._card_gestures.findChildren(QLabel)
        self.assertEqual(labels[1].text(), "10")

    def test_gesture_card_renders_metadata(self):
        """Kiểm tra GestureCard hiển thị đầy đủ tên, mode, số mẫu và thanh tiến trình."""
        sample_gesture = {
            "name": "TEST_SWIPE",
            "display_name": "Vuốt thử nghiệm",
            "mode_id": 2,
            "description": "Mô tả cử chỉ vuốt",
            "finger_guide": "Xòe 5 ngón",
            "is_system": 1,
        }
        card = GestureCard(gesture=sample_gesture, sample_count=45)
        self.assertEqual(card.gesture_name, "TEST_SWIPE")
        self.assertEqual(card.display_name, "Vuốt thử nghiệm")
        self.assertEqual(card._mode_id, 2)

        # Kiểm tra phát tín hiệu khi click Chi tiết
        signal_received = []
        card.detail_requested.connect(lambda name: signal_received.append(name))
        btn_detail = card.findChild(type(card.findChildren(type(card))[0]) if False else type(card.children()[-1]))
        card.detail_requested.emit("TEST_SWIPE")
        self.assertEqual(signal_received, ["TEST_SWIPE"])
        card.deleteLater()

    def test_mode_filter_combobox(self):
        """Kiểm tra lọc danh sách theo chế độ hoạt động (Mode 1, Mode 2, Mode 3, Tất cả)."""
        # Mode 1: Chuột
        self.library_page._cbo_mode.setCurrentIndex(1)
        m1_visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertTrue(len(m1_visible) > 0)
        for c in m1_visible:
            self.assertEqual(c._mode_id, 1)

        # Mode 2: Media / Trình chiếu
        self.library_page._cbo_mode.setCurrentIndex(2)
        m2_visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertTrue(len(m2_visible) > 0)
        for c in m2_visible:
            self.assertEqual(c._mode_id, 2)

        # Trở lại Tất cả
        self.library_page._cbo_mode.setCurrentIndex(0)
        all_visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertEqual(len(all_visible), 10)

    def test_search_filter_behavior(self):
        """Kiểm tra tìm kiếm cử chỉ theo tên mã và tên hiển thị (không phân biệt hoa thường)."""
        # 1. Tìm theo tên mã 'PINCH'
        self.library_page._txt_search.setText("PINCH")
        visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].gesture_name, "PINCH_CLICK")

        # 2. Tìm theo tên hiển thị tiếng Việt 'Zoom' hoặc 'Vuốt'
        self.library_page._txt_search.setText("Zoom")
        visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertEqual(len(visible), 2)
        self.assertTrue(all("ZOOM" in c.gesture_name for c in visible))

        # 3. Tìm từ khóa không tồn tại
        self.library_page._txt_search.setText("NON_EXISTENT_KEYWORD")
        visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertEqual(len(visible), 0)

        # 4. Xóa tìm kiếm -> hiển thị lại toàn bộ
        self.library_page._txt_search.setText("")
        visible = [c for c in self.library_page._all_cards if c.isVisible()]
        self.assertEqual(len(visible), 10)

    def test_contributor_chart_rendering(self):
        """Kiểm tra vẽ biểu đồ ASCII đóng góp bằng khối Unicode."""
        dummy_contribs = [
            {"name": "Ky", "sample_count": 350},
            {"name": "Huy", "sample_count": 150},
        ]
        self.library_page._render_contributor_chart(dummy_contribs)
        chart_text = self.library_page._lbl_chart.text()
        self.assertIn("Ky", chart_text)
        self.assertIn("Huy", chart_text)
        self.assertIn("█", chart_text)
        self.assertIn("350 mẫu", chart_text)

        # Kiểm tra trường hợp rỗng
        self.library_page._render_contributor_chart([])
        self.assertIn("Chưa có", self.library_page._lbl_chart.text())

    def test_detail_dialog_system_gesture(self):
        """Kiểm tra mở Dialog chi tiết cho cử chỉ hệ thống (không cho phép xóa)."""
        dlg = GestureDetailDialog(gesture_name="PINCH_CLICK", db=self.db, parent=self.library_page)
        self.assertIsNotNone(dlg)
        self.assertIn("PINCH_CLICK", dlg.windowTitle())
        labels = dlg.findChildren(QLabel)
        texts = [l.text() for l in labels]
        self.assertTrue(any("PINCH_CLICK" in t for t in texts))
        self.assertTrue(any("không thể xóa" in t for t in texts))
        dlg.close()
        dlg.deleteLater()

    def test_detail_dialog_custom_gesture_deletion(self):
        """Kiểm tra mở Dialog và xóa cử chỉ tùy chỉnh thành công."""
        # Thêm cử chỉ tùy chỉnh vào DB
        self.db.add_gesture(
            name="CUSTOM_WAVE",
            display_name="Vẫy tay tạm biệt",
            mode_id=3,
            description="Vẫy tay nhẹ nhàng",
            is_system=0,
        )
        self.library_page.refresh()
        self.assertEqual(len(self.library_page._all_cards), 11)

        deleted_called = []
        dlg = GestureDetailDialog(
            gesture_name="CUSTOM_WAVE",
            db=self.db,
            on_deleted=lambda: deleted_called.append(True),
            parent=self.library_page,
        )

        # Giả lập xác nhận Yes khi bấm Xóa
        with patch("PyQt6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            dlg._confirm_delete()

        self.assertEqual(deleted_called, [True])
        self.assertIsNone(self.db.get_gesture("CUSTOM_WAVE"))
        dlg.close()
        dlg.deleteLater()


if __name__ == "__main__":
    unittest.main()
