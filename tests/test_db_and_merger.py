"""
tests/test_db_and_merger.py
===========================
Kiểm thử tự động cho hệ thống Database SQLite và Bộ nạp dữ liệu phân tán.
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database.db_manager import DatabaseManager
from src.database.dataset_merger import (
    save_single_sample,
    sync_folder_with_db,
    load_merged_dataset,
    export_contributor_zip,
    import_contributor_zip,
)


class TestDatabaseManager(unittest.TestCase):
    """Kiểm thử DatabaseManager: khởi tạo, thêm cử chỉ, quản lý mẫu, cài đặt."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_gestures.db"
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_gestures_initialized(self):
        gestures = self.db.get_all_gestures()
        self.assertGreaterEqual(len(gestures), 10)
        names = [g["name"] for g in gestures]
        self.assertIn("PINCH_CLICK", names)
        self.assertIn("IDLE", names)
        self.assertIn("FIST_LOCK", names)

    def test_add_and_delete_custom_gesture(self):
        # Thêm cử chỉ tùy chỉnh
        ok = self.db.add_gesture(
            name="CUSTOM_WAVE",
            display_name="Vẫy tay chào",
            mode_id=2,
            description="Cử chỉ vẫy tay chào bạn bè",
            finger_guide="Xòe 5 ngón lắc qua lại",
            is_system=0,
        )
        self.assertTrue(ok)

        g = self.db.get_gesture("CUSTOM_WAVE")
        self.assertIsNotNone(g)
        self.assertEqual(g["display_name"], "Vẫy tay chào")

        # Cử chỉ hệ thống không thể xóa
        self.assertFalse(self.db.delete_gesture("PINCH_CLICK"))

        # Cử chỉ tùy chỉnh có thể xóa
        deleted = self.db.delete_gesture("CUSTOM_WAVE")
        self.assertTrue(deleted)
        self.assertIsNone(self.db.get_gesture("CUSTOM_WAVE"))

    def test_add_sample_and_contributor_tracking(self):
        fake_file = str(Path(self.temp_dir) / "dummy_sample.npz")
        ok = self.db.add_sample(
            gesture_name="PINCH_CLICK",
            file_path=fake_file,
            author="Huy",
            num_frames=45,
            feature_dim=158,
        )
        self.assertTrue(ok)

        # Kiểm tra sample count
        counts = self.db.get_gesture_sample_counts()
        self.assertEqual(counts["PINCH_CLICK"], 1)

        # Kiểm tra contributor
        contribs = self.db.get_contributors()
        self.assertTrue(any(c["name"] == "Huy" and c["sample_count"] >= 1 for c in contribs))

    def test_app_settings(self):
        self.db.set_setting("mouse_sensitivity", "1.75")
        val = self.db.get_setting("mouse_sensitivity")
        self.assertEqual(val, "1.75")

        # Đọc key không tồn tại
        fallback = self.db.get_setting("non_existent_key", default="default_val")
        self.assertEqual(fallback, "default_val")


class TestDatasetMerger(unittest.TestCase):
    """Kiểm thử DatasetMerger: lưu mẫu đơn lẻ, đồng bộ thư mục, xuất/nhập ZIP, gộp tensor."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_gestures.db"
        self.db = DatabaseManager(self.db_path)
        self.seq_dir = Path(self.temp_dir) / "sequences"
        self.seq_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_single_sample_and_sync(self):
        dummy_seq = np.random.randn(45, 158).astype(np.float32)
        saved_path = save_single_sample(
            sequence=dummy_seq,
            gesture_name="SWIPE_LEFT",
            author="Nam",
            output_dir=self.seq_dir,
            db=self.db,
        )
        self.assertTrue(saved_path.exists())

        # Kiểm tra đồng bộ lại thư mục
        synced = sync_folder_with_db(self.db, sequences_dir=self.seq_dir)
        # Vì đã add lúc save nên sync sẽ phát hiện 0 mẫu mới
        self.assertEqual(synced, 0)

        counts = self.db.get_gesture_sample_counts()
        self.assertEqual(counts["SWIPE_LEFT"], 1)

    def test_load_merged_dataset(self):
        # Tạo 3 mẫu cho 2 cử chỉ khác nhau
        for i in range(2):
            save_single_sample(
                sequence=np.ones((45, 158), dtype=np.float32) * i,
                gesture_name="PINCH_CLICK",
                author="Ky",
                output_dir=self.seq_dir,
                db=self.db,
            )
        save_single_sample(
            sequence=np.ones((45, 158), dtype=np.float32) * 5,
            gesture_name="SWIPE_RIGHT",
            author="Huy",
            output_dir=self.seq_dir,
            db=self.db,
        )

        X, y, classes = load_merged_dataset(
            db=self.db,
            sequences_dir=self.seq_dir,
            include_legacy=False,
            target_seq_len=45,
            target_dim=158,
        )

        self.assertEqual(X.shape, (3, 45, 158))
        self.assertEqual(len(y), 3)
        self.assertIn("PINCH_CLICK", classes)
        self.assertIn("SWIPE_RIGHT", classes)

    def test_export_and_import_zip(self):
        # Lưu 2 mẫu của "Huy" và 1 mẫu của "Nam"
        save_single_sample(np.zeros((45, 158)), "PINCH_CLICK", "Huy", self.seq_dir, self.db)
        save_single_sample(np.zeros((45, 158)), "SWIPE_LEFT", "Huy", self.seq_dir, self.db)
        save_single_sample(np.zeros((45, 158)), "SWIPE_RIGHT", "Nam", self.seq_dir, self.db)

        # Xuất gói ZIP của Huy
        zip_path = Path(self.temp_dir) / "huy_dataset.zip"
        n_exported = export_contributor_zip("Huy", zip_path, self.seq_dir)
        self.assertEqual(n_exported, 2)
        self.assertTrue(zip_path.exists())

        # Nhập vào thư mục đích khác
        import_dest = Path(self.temp_dir) / "imported_sequences"
        new_db = DatabaseManager(Path(self.temp_dir) / "import_test.db")
        n_imported = import_contributor_zip(zip_path, import_dest, new_db)
        self.assertEqual(n_imported, 2)

        counts = new_db.get_gesture_sample_counts()
        self.assertEqual(counts["PINCH_CLICK"], 1)
        self.assertEqual(counts["SWIPE_LEFT"], 1)


if __name__ == "__main__":
    unittest.main()
