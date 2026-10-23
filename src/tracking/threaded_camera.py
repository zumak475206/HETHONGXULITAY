"""
src/tracking/threaded_camera.py
===============================
Camera đọc đa luồng (Zero-Buffer-Lag Threaded VideoCapture).
Khởi chạy một daemon thread chuyên biệt liên tục đọc frames từ cv2.VideoCapture
với tốc độ phần cứng tối đa, luôn lưu trữ frame mới nhất và ghi đè (single-latch)
để loại bỏ hoàn toàn hiện tượng trễ đệm DirectShow/OpenCV.

Lợi ích:
  - CAP_PROP_BUFFERSIZE = 1 triệt tiêu DirectShow Ring Buffer 4 frames (~133ms).
  - Single-latch atomic lock: Luôn trả về frame tươi mới nhất tại thời điểm gọi.
  - Tự động fallback cv2.CAP_DSHOW trên Windows để mở camera nhanh và chuẩn phân giải.
"""

import sys
import time
import threading
from typing import Optional, Tuple, Any, Union
import cv2
import numpy as np


class ThreadedCamera:
    """
    Camera đọc đa luồng độc lập (Zero-Buffer-Lag Threaded VideoCapture).
    """

    def __init__(
        self,
        src: Union[int, str, Any] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        buffersize: int = 1,
    ):
        """
        Khởi tạo và bắt đầu luồng đọc camera chuyên biệt.

        Tham số
        -------
        src : int or str or cv2.VideoCapture
            Camera index (0, 1, ...), đường dẫn video/RTSP, hoặc đối tượng cv2.VideoCapture / Mock.
        width : int, optional
            Chiều rộng khung hình (CAMERA_WIDTH).
        height : int, optional
            Chiều cao khung hình (CAMERA_HEIGHT).
        fps : int, optional
            Tốc độ khung hình yêu cầu.
        buffersize : int, default 1
            Kích thước bộ đệm phần cứng OpenCV (CAP_PROP_BUFFERSIZE = 1).
        """
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.buffersize = buffersize

        # Hỗ trợ truyền mock hoặc đối tượng VideoCapture đã có sẵn
        if hasattr(src, "read") and hasattr(src, "release") and not isinstance(src, (int, str)):
            self.cap = src
        else:
            if sys.platform == "win32" and isinstance(src, int):
                self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback sang backend mặc định nếu CAP_DSHOW không mở được
                    self.cap = cv2.VideoCapture(src)
            else:
                self.cap = cv2.VideoCapture(src)

        # Cài đặt thuộc tính phần cứng
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffersize)))
        except Exception:
            pass

        self._lock = threading.Lock()
        self._stopped = False
        self._ret = False
        self._latest_frame: Optional[np.ndarray] = None

        # Đọc thử frame đầu tiên một cách đồng bộ để xác nhận camera hoạt động
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._ret = True
                self._latest_frame = frame

        self._thread = threading.Thread(
            target=self._worker,
            name="ThreadedCameraWorker",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        """Vòng lặp ngầm liên tục đọc frame từ phần cứng và ghi đè single latch."""
        while not self._stopped:
            try:
                if not self.cap.isOpened():
                    time.sleep(0.01)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    with self._lock:
                        self._ret = False
                    time.sleep(0.005)
                    continue

                with self._lock:
                    self._ret = True
                    self._latest_frame = frame
            except Exception:
                with self._lock:
                    self._ret = False
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lấy frame mới nhất hiện có từ camera mà không bao giờ bị nghẽn queue.
        Trả về: (ret: bool, frame: np.ndarray or None)
        Nếu camera dừng hoặc đọc thất bại, luôn trả về (False, None).
        """
        with self._lock:
            if self._stopped or not self._ret or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def isOpened(self) -> bool:
        """Kiểm tra camera có đang mở hay không."""
        return bool(self.cap.isOpened() and not self._stopped)

    def release(self) -> None:
        """Dừng worker thread và giải phóng webcam sạch sẽ."""
        self._stopped = True
        with self._lock:
            self._ret = False
            self._latest_frame = None
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if hasattr(self, "cap") and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def set(self, prop_id: int, val: Any) -> bool:
        """Đặt thuộc tính cho VideoCapture."""
        return bool(self.cap.set(prop_id, val))

    def get(self, prop_id: int) -> Any:
        """Lấy giá trị thuộc tính từ VideoCapture."""
        return self.cap.get(prop_id)

    def __enter__(self) -> "ThreadedCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
