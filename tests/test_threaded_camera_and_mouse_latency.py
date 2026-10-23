"""
tests/test_threaded_camera_and_mouse_latency.py
================================================
Kiểm thử tự động cho:
1. ThreadedCamera: Đọc đa luồng, single-latch atomic lock, CAP_PROP_BUFFERSIZE = 1, context manager và giải phóng an toàn.
2. optimize_windows_process: Nâng priority tiến trình lên ABOVE_NORMAL và tắt Windows EcoQoS / Power Throttling.
3. Phân tích & Đo đạc độ trễ bước nhảy (Step Response Latency):
   Chứng minh toán học và thực nghiệm việc tách riêng đường truyền Mode 1 (Raw landmarks + One Euro Filter Beta=0.04)
   giải phóng hoàn toàn con trỏ chuột khỏi hiện tượng trễ ~170ms do bộ lọc Kalman kép gây ra.
"""

import unittest
from unittest.mock import MagicMock, patch
import time
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.tracking.threaded_camera import ThreadedCamera
from src.utils.process_optimizer import optimize_windows_process
from src.tracking.kalman_filter import HandKalmanFilter
from src.tracking.one_euro_filter import PointFilter
from src.modes.mode1_mouse import MouseController


class TestThreadedCameraAndMouseLatency(unittest.TestCase):

    def test_threaded_camera_lifecycle_and_single_latch(self):
        """Kiểm tra ThreadedCamera khởi tạo, set buffer 1, đọc single latch và giải phóng an toàn."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)

        cam = ThreadedCamera(
            src=mock_cap,
            width=1280,
            height=720,
            fps=30,
            buffersize=1,
        )

        # 1. Xác nhận các thuộc tính được cài đặt chính xác
        mock_cap.set.assert_any_call(config.cv2.CAP_PROP_FRAME_WIDTH if hasattr(config, "cv2") else 3, 1280)
        mock_cap.set.assert_any_call(config.cv2.CAP_PROP_FRAME_HEIGHT if hasattr(config, "cv2") else 4, 720)
        self.assertTrue(cam.isOpened())

        # 2. Đọc frame từ camera đa luồng
        time.sleep(0.05)  # Chờ worker thread nạp frame
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))

        # 3. Giải phóng an toàn
        cam.release()
        self.assertFalse(cam.isOpened())
        self.assertTrue(mock_cap.release.called)

    def test_threaded_camera_context_manager_and_failed_read(self):
        """Kiểm tra ThreadedCamera hoạt động với context manager (with) và xử lý lỗi camera read an toàn."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with ThreadedCamera(src=mock_cap, buffersize=1) as cam:
            self.assertTrue(cam.isOpened())
            time.sleep(0.02)
            ret, frame = cam.read()
            self.assertFalse(ret)
            self.assertIsNone(frame)

        # Ra khỏi context manager -> Tự động release
        self.assertTrue(mock_cap.release.called)
        self.assertFalse(cam.isOpened())

    def test_threaded_camera_read_after_release_returns_false_none(self):
        """Kiểm tra ThreadedCamera sau khi release() thì read() luôn trả về (False, None) an toàn."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))

        cam = ThreadedCamera(src=mock_cap, buffersize=1)
        time.sleep(0.02)
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)

        # Release camera
        cam.release()
        self.assertFalse(cam.isOpened())

        # Sau khi release, tuyệt đối không được trả về (True, None) hoặc frame cũ
        ret_after, frame_after = cam.read()
        self.assertFalse(ret_after, "ret sau khi release phai la False")
        self.assertIsNone(frame_after, "frame sau khi release phai la None")

    def test_mouse_controller_dynamic_resolution_adaptation(self):
        """Kiểm tra MouseController tự động thích ứng kích thước khung hình camera khi phần cứng đổi phân giải."""
        controller = MouseController(frame_w=1280, frame_h=720)
        self.assertEqual(controller.frame_w, 1280)
        self.assertEqual(controller.frame_h, 720)
        self.assertEqual(controller.margin_x, int(1280 * 0.25))
        self.assertEqual(controller.margin_y, int(720 * 0.20))

        # Camera thực tế trả về 640x480
        frame_480p = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.process(frame_480p, [])

        # Kiểm tra tự động thích ứng
        self.assertEqual(controller.frame_w, 640)
        self.assertEqual(controller.frame_h, 480)
        self.assertEqual(controller.margin_x, int(640 * 0.25))
        self.assertEqual(controller.margin_y, int(480 * 0.20))

    def test_optimize_windows_process_execution(self):
        """Kiểm tra optimize_windows_process() thực thi thành công trên Windows."""
        res = optimize_windows_process()
        self.assertIsInstance(res, dict)
        self.assertIn("priority_set", res)
        self.assertIn("throttling_disabled", res)
        if sys.platform == "win32":
            self.assertTrue(res["priority_set"], "Priority phai duoc set thanh cong tren Windows")
            self.assertTrue(res["throttling_disabled"], "Power Throttling phai duoc tat thanh cong tren Windows")

    def test_mouse_step_response_latency_comparison(self):
        """
        Benchmark đối chiếu độ trễ bước nhảy (Step Response Latency):
        So sánh Pipeline A cũ (Kalman Q=0.05, R=0.80 -> One Euro Filter Beta=0.008)
        với Pipeline B mới (Direct Raw Landmarks -> One Euro Filter Beta=0.04).
        """
        kf = HandKalmanFilter(dt=1.0/30.0, process_noise_var=0.05, measurement_noise_var=0.80)
        filter_old = PointFilter(min_cutoff=1.0, beta=0.008)
        filter_new = PointFilter(min_cutoff=1.0, beta=config.MOUSE_ONE_EURO_BETA)

        # 1. Ổn định tại vị trí ban đầu x = 300.0
        t = 0.0
        for _ in range(15):
            t += 1.0/30.0
            raw_pts = [(i, 300.0, 300.0, 0.0) for i in range(21)]
            s_pts, _ = kf.update(raw_pts)
            filter_old.filter(s_pts[8][1], s_pts[8][2], timestamp=t)
            filter_new.filter(raw_pts[8][1], raw_pts[8][2], timestamp=t)

        # 2. Tạo bước nhảy đột ngột (Step Jump) từ x=300.0 sang x=600.0 (delta = 300px)
        # Đo đạc frame 1 (tức thì sau 33.3ms)
        t += 1.0/30.0
        step_raw = [(i, 600.0, 300.0, 0.0) for i in range(21)]
        s_step, _ = kf.update(step_raw)
        old_x_f1, _ = filter_old.filter(s_step[8][1], s_step[8][2], timestamp=t)
        new_x_f1, _ = filter_new.filter(step_raw[8][1], step_raw[8][2], timestamp=t)

        pct_old_f1 = (old_x_f1 - 300.0) / 300.0
        pct_new_f1 = (new_x_f1 - 300.0) / 300.0

        # Đo đạc frame 2 (sau 66.6ms)
        t += 1.0/30.0
        s_step2, _ = kf.update(step_raw)
        old_x_f2, _ = filter_old.filter(s_step2[8][1], s_step2[8][2], timestamp=t)
        new_x_f2, _ = filter_new.filter(step_raw[8][1], step_raw[8][2], timestamp=t)

        pct_old_f2 = (old_x_f2 - 300.0) / 300.0
        pct_new_f2 = (new_x_f2 - 300.0) / 300.0

        print(f"\n[BENCHMARK STEP RESPONSE]")
        print(f"  - Frame 1: Old (Kalman+OneEuro) = {pct_old_f1*100:.1f}% | New (Raw+OneEuro) = {pct_new_f1*100:.1f}%")
        print(f"  - Frame 2: Old (Kalman+OneEuro) = {pct_old_f2*100:.1f}% | New (Raw+OneEuro) = {pct_new_f2*100:.1f}%")

        # Khẳng định chất lượng:
        # Pipeline mới bắt kịp > 90% ngay ở Frame 1, trong khi pipeline cũ chỉ đạt < 10%
        self.assertGreater(pct_new_f1, 0.85, "Pipeline moi phai bat kip >= 85% ngay o frame dau tien")
        self.assertLess(pct_old_f1, 0.20, "Pipeline cu bi Kalman tri hoan nang ne nen < 20% o frame 1")
        self.assertGreater(pct_new_f2, 0.95, "Pipeline moi phai dat >= 95% o frame thu 2 (Zero-Lag)")


if __name__ == "__main__":
    unittest.main()
