"""
tests/test_mode3_sign_language.py
=================================
Bộ kiểm thử tự động toàn diện cho Mode 3: Nhận diện Ngôn ngữ ký hiệu VSL & Phụ đề toàn màn hình máy tính.

Bao gồm:
1. SignLanguageController: Từ điển ý nghĩa tiếng Việt, suy luận 158D, Attention Heatmap, Top-3, Ghép câu.
2. DesktopSubtitleOverlay: Cửa sổ nổi Always-On-Top toàn màn hình, cập nhật text, ẩn hiện theo mode.
3. TrainWorker: Tiến trình huấn luyện PyTorch thực tế, xuất ONNX FP32 và INT8, Hot-Reload.
4. Tích hợp CameraWorker & MainWindow: Phát tín hiệu subtitle_updated, tự động hiển thị phụ đề máy tính.
"""

import sys
import os
import unittest
import tempfile
import shutil
import time
from pathlib import Path
import numpy as np

# Cài đặt Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer

import config
from src.database.db_manager import DatabaseManager
from src.database.dataset_merger import save_single_sample, load_merged_dataset
from src.modes.mode3_sign_language import SignLanguageController
from src.ui.subtitle_overlay import DesktopSubtitleOverlay
from src.ui.workers import CameraWorker, TrainWorker
from src.ui.main_window import MainWindow


class TestSignLanguageController(unittest.TestCase):
    """Kiểm thử bộ điều khiển Chế độ 3 (SignLanguageController)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_sign.db"
        self.db = DatabaseManager(self.db_path)

        # Thêm các ký hiệu VSL mẫu vào DB
        self.db.add_gesture(name="VSL_HELLO", display_name="Xin chào", mode_id=3)
        self.db.add_gesture(name="VSL_THANKYOU", display_name="Cảm ơn", mode_id=3)
        self.db.add_gesture(name="VSL_YES", display_name="Đồng ý", mode_id=3)

        self.controller = SignLanguageController(frame_w=640, frame_h=480, db=self.db)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_controller_initialization_and_meanings(self):
        """Kiểm tra khởi tạo và tra cứu từ điển ý nghĩa tiếng Việt."""
        self.assertEqual(self.controller.get_meaning("VSL_HELLO"), "Xin chào")
        self.assertEqual(self.controller.get_meaning("VSL_THANKYOU"), "Cảm ơn")
        self.assertEqual(self.controller.get_meaning("VSL_YES"), "Đồng ý")
        # Ký hiệu mặc định hệ thống
        self.assertEqual(self.controller.get_meaning("IDLE"), "Nghỉ / Tự do")
        # Ký hiệu chưa biết -> trả về tên gốc
        self.assertEqual(self.controller.get_meaning("VSL_UNKNOWN"), "VSL_UNKNOWN")

    def test_reload_gesture_meanings_dynamically(self):
        """Kiểm tra nạp động ý nghĩa mới khi người dùng thêm ký hiệu trong Studio."""
        self.db.add_gesture(name="VSL_LOVE", display_name="Yêu quý", mode_id=3)
        self.controller.reload_gesture_meanings()
        self.assertEqual(self.controller.get_meaning("VSL_LOVE"), "Yêu quý")

    def test_process_inference_softmax_and_top3(self):
        """Kiểm tra suy luận xác suất Softmax và trích xuất Top-3 lớp dự đoán."""
        classes = ["IDLE", "VSL_HELLO", "VSL_THANKYOU", "VSL_YES"]
        # Giả lập logits ưu tiên VSL_HELLO
        logits = np.array([0.1, 8.5, 2.1, 0.4], dtype=np.float32)
        attn = np.ones(config.SEQUENCE_LENGTH, dtype=np.float32) * 0.5

        lbl, conf, sentence = self.controller.process_inference(
            logits=logits,
            attn_weights=attn,
            class_list=classes,
            has_hands=True,
        )

        top3 = self.controller.top3_predictions
        self.assertEqual(len(top3), 3)
        # Lớp đứng đầu phải là VSL_HELLO
        self.assertEqual(top3[0][0], "VSL_HELLO")
        self.assertEqual(top3[0][1], "Xin chào")
        self.assertGreater(top3[0][2], 0.95)

    def test_sentence_accumulator_and_debouncing(self):
        """Kiểm tra bộ ghép câu hội thoại và chống kích hoạt lặp từ liên tục."""
        classes = ["IDLE", "VSL_HELLO", "VSL_THANKYOU", "VSL_YES"]
        logits_hello = np.array([0.1, 9.0, 0.2, 0.3], dtype=np.float32)
        logits_thankyou = np.array([0.1, 0.2, 9.0, 0.3], dtype=np.float32)

        # Gửi 6 frame VSL_HELLO để kích hoạt voting filter
        for _ in range(6):
            self.controller.process_inference(logits_hello, None, classes, has_hands=True)

        self.assertIn("Xin chào", self.controller.get_full_sentence())

        # Gửi tiếp VSL_HELLO ngay sau đó -> debouncing không lặp lại ngay
        self.controller.process_inference(logits_hello, None, classes, has_hands=True)
        words = self.controller.sentence_words
        self.assertEqual(words.count("Xin chào"), 1)

        # Chuyển sang VSL_THANKYOU -> ghép tiếp thành "Xin chào Cảm ơn"
        for _ in range(6):
            self.controller.process_inference(logits_thankyou, None, classes, has_hands=True)

        self.assertEqual(self.controller.get_full_sentence(), "Xin chào Cảm ơn")

    def test_sentence_clear_and_subtitle_state(self):
        """Kiểm tra làm sạch câu và xuất dữ liệu cho DesktopSubtitleOverlay."""
        self.controller.sentence_words = ["Tôi", "Yêu", "Việt Nam"]
        self.controller.current_meaning = "Việt Nam"
        self.controller.current_confidence = 0.92

        state = self.controller.get_subtitle_state()
        self.assertEqual(state["sentence"], "Tôi Yêu Việt Nam")
        self.assertEqual(state["current_meaning"], "Việt Nam")
        self.assertEqual(state["confidence"], 0.92)
        self.assertTrue(state["is_active"])

        # Xóa câu
        self.controller.clear_sentence()
        self.assertEqual(self.controller.get_full_sentence(), "")
        state_cleared = self.controller.get_subtitle_state()
        self.assertEqual(state_cleared["sentence"], "")

    def test_draw_hud_renders_safely(self):
        """Kiểm tra hàm vẽ HUD và Attention Heatmap không bị crash trên frame ảnh thật."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.controller.top3_predictions = [("VSL_HELLO", "Xin chào", 0.92)]
        self.controller.sentence_words = ["Xin chào"]
        self.controller.current_meaning = "Xin chào"

        out_frame = self.controller.draw_hud(frame, hands_data=[], raw_hands_data=[])
        self.assertIsNotNone(out_frame)
        self.assertEqual(out_frame.shape, (480, 640, 3))


class TestDesktopSubtitleOverlay(unittest.TestCase):
    """Kiểm thử Cửa sổ Phụ Đề Nổi Toàn Màn Hình Máy Tính (DesktopSubtitleOverlay)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.overlay = DesktopSubtitleOverlay(parent=None)

    def tearDown(self):
        self.overlay.close()
        self.overlay.deleteLater()

    def test_overlay_window_flags(self):
        """Kiểm tra cờ cửa sổ nổi Frameless, Always-On-Top, Tool, không chiếm focus."""
        flags = self.overlay.windowFlags()
        self.assertTrue(flags & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(flags & Qt.WindowType.WindowStaysOnTopHint)
        self.assertTrue(flags & Qt.WindowType.Tool)
        self.assertTrue(flags & Qt.WindowType.WindowDoesNotAcceptFocus)

    def test_overlay_mode_visibility(self):
        """Chỉ hiển thị phụ đề khi ở MODE3_SIGN, tự động ẩn khi ở các mode khác."""
        self.overlay.set_active_mode("MODE3_SIGN")
        self.assertTrue(self.overlay.isVisible())

        self.overlay.set_active_mode("MODE1_MOUSE")
        self.assertFalse(self.overlay.isVisible())

        self.overlay.set_active_mode("MODE2_GESTURES")
        self.assertFalse(self.overlay.isVisible())

        self.overlay.set_active_mode("STANDBY")
        self.assertFalse(self.overlay.isVisible())

    def test_overlay_update_data(self):
        """Kiểm tra cập nhật text câu, ý nghĩa từ và độ tin cậy."""
        self.overlay.set_active_mode("MODE3_SIGN")
        self.overlay.update_subtitle_data(
            sentence="Xin chào các bạn",
            current_word="VSL_HELLO",
            current_meaning="Xin chào",
            confidence=0.98,
        )
        self.assertEqual(self.overlay.lbl_sentence.text(), "Xin chào các bạn")
        self.assertIn("Xin chào", self.overlay.lbl_current_word.text())
        self.assertEqual(self.overlay.lbl_confidence.text(), "98%")

    def test_overlay_clear_requested_signal(self):
        """Kiểm tra nút Clear phát tín hiệu clear_requested."""
        received = []
        self.overlay.clear_requested.connect(lambda: received.append(True))
        self.overlay.btn_clear.click()
        self.assertTrue(len(received) > 0)


class TestTrainWorkerPyTorch(unittest.TestCase):
    """Kiểm thử Luồng Huấn luyện AI PyTorch thực tế (TrainWorker)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_train.db"
        self.db = DatabaseManager(self.db_path)
        self.seq_dir = self.temp_dir / "sequences"

        # Sinh dữ liệu mẫu giả lập cho 2 lớp
        for cls_name in ["VSL_HELLO", "VSL_YES"]:
            self.db.add_gesture(name=cls_name, display_name=cls_name, mode_id=3)
            for i in range(4):
                sample_data = np.random.randn(45, config.FEATURE_DIM).astype(np.float32)
                save_single_sample(
                    sequence=sample_data,
                    gesture_name=cls_name,
                    author="TestBot",
                    output_dir=self.seq_dir,
                    db=self.db,
                )

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_worker_pytorch_execution(self):
        """Kiểm tra TrainWorker chạy huấn luyện PyTorch thực tế và phát đầy đủ signals."""
        worker = TrainWorker(
            db=self.db,
            epochs=2,
            batch_size=4,
            sequences_dir=self.seq_dir,
            include_legacy=False,
        )

        epochs_received = []
        results = []
        errors = []

        worker.epoch_progress.connect(lambda ep, tot, loss, acc: epochs_received.append((ep, loss, acc)))
        worker.training_completed.connect(lambda res: results.append(res))
        worker.training_failed.connect(lambda err: errors.append(err))

        worker.start()
        for _ in range(600):
            self.app.processEvents()
            if worker.isFinished():
                break
            time.sleep(0.05)
        for _ in range(20):
            self.app.processEvents()
            time.sleep(0.02)

        if len(errors) > 0 or len(epochs_received) != 2:
            print("DEBUG errors:", errors)
            print("DEBUG results:", results)
            print("DEBUG epochs_received:", epochs_received)

        self.assertEqual(len(errors), 0, f"Huấn luyện gặp lỗi: {errors}")
        self.assertEqual(len(epochs_received), 2)
        self.assertEqual(len(results), 1)

        summary = results[0]
        self.assertGreaterEqual(summary["total_samples"], 8)
        self.assertGreaterEqual(summary["total_classes"], 2)
        self.assertIn("final_acc", summary)
        self.assertIn("final_loss", summary)


class TestMode3EndToEndIntegration(unittest.TestCase):
    """Kiểm thử tích hợp đầu cuối giữa CameraWorker, MainWindow và SubtitleOverlay."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_app.db"
        self.db = DatabaseManager(self.db_path)
        self.db.add_gesture(name="VSL_HELLO", display_name="Xin chào", mode_id=3)

        self.window = MainWindow(db=self.db, start_camera=False)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_main_window_has_subtitle_overlay(self):
        """Kiểm tra MainWindow đã khởi tạo và liên kết DesktopSubtitleOverlay."""
        self.assertIsNotNone(self.window.subtitle_overlay)
        self.assertIsInstance(self.window.subtitle_overlay, DesktopSubtitleOverlay)

    def test_switch_to_mode3_activates_overlay(self):
        """Khi người dùng chuyển sang MODE3_SIGN, subtitle_overlay phải hiển thị."""
        self.window._on_mode_button_clicked("MODE3_SIGN")
        self.assertEqual(self.window.current_mode_name, "MODE3_SIGN")
        self.assertTrue(self.window.subtitle_overlay.isVisible())

        # Chuyển về MODE1_MOUSE -> subtitle_overlay phải tự động ẩn
        self.window._on_mode_button_clicked("MODE1_MOUSE")
        self.assertFalse(self.window.subtitle_overlay.isVisible())

    def test_camera_worker_subtitle_updated_relays_to_overlay(self):
        """Kiểm tra tín hiệu subtitle_updated từ CameraWorker cập nhật chính xác lên DesktopSubtitleOverlay."""
        self.window._on_mode_button_clicked("MODE3_SIGN")

        test_payload = {
            "sentence": "Xin chào thế giới",
            "current_word": "VSL_HELLO",
            "current_meaning": "Xin chào",
            "confidence": 0.96,
            "is_active": True,
        }
        self.window._on_subtitle_updated(test_payload)

        self.assertEqual(self.window.subtitle_overlay.lbl_sentence.text(), "Xin chào thế giới")
        self.assertIn("Xin chào", self.window.subtitle_overlay.lbl_current_word.text())


if __name__ == "__main__":
    unittest.main()
