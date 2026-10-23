"""
tools/sequence_recorder.py
==========================
Công cụ thu thập dữ liệu chuỗi cử chỉ thời gian thực tương tác qua OpenCV.

Chức năng:
1. Giao diện trực quan hỗ trợ chọn cử chỉ mục tiêu bằng phím số (0-9).
2. Nhấn 'R': Bắt đầu đếm ngược 3-2-1 trên màn hình và tự động thu chính xác 30 frame đặc trưng 157D.
3. Nhấn 'Space': Chế độ Playback - Phát lại chuỗi vừa quay dưới dạng video khung xương để bạn kiểm tra chất lượng thao tác.
4. Nhấn 'Z': Xóa ngay chuỗi vừa quay nếu bị vấp hoặc sai động tác.
5. Nhấn 'S': Lưu toàn bộ mẫu đã thu thập vào dataset/gesture_sequences.npz.
6. Hiển thị thống kê số lượng mẫu của từng nhãn trực tiếp trên HUD.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Đảm bảo import được config và các module từ src
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing.clahe_enhancer import CLAHEEnhancer
from src.tracking.kalman_filter import HandKalmanFilter
from src.features.feature_extractor import extract_158d_features, extract_157d_features, extract_112d_features

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    mp_hands = None
    mp_drawing = None


class SequenceRecorder:
    """Ứng dụng OpenCV tương tác thu thập tập dữ liệu chuỗi 30 frame."""

    def __init__(self):
        self.enhancer = CLAHEEnhancer(enabled=config.CLAHE_ENABLED)
        self.kalman_hands = [HandKalmanFilter(), HandKalmanFilter()]
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        self.mp_hand_detector = mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        ) if mp_hands else None

        self.current_class_idx = 1  # Mặc định PINCH_CLICK
        self.classes = config.GESTURE_CLASSES

        # Bộ nhớ lưu trữ mẫu trong phiên làm việc
        self.dataset_file = config.DATASET_DIR / "gesture_sequences.npz"
        self.recorded_sequences: List[np.ndarray] = []
        self.recorded_labels: List[int] = []

        # Tải dữ liệu đã có trước đó nếu tồn tại
        if self.dataset_file.exists():
            try:
                data = np.load(self.dataset_file)
                raw_X = data["X"]
                self.recorded_labels = list(data["y"])
                if len(raw_X) > 0:
                    from src.training.train import convert_to_158d, resample_sequence_length
                    if raw_X.shape[-1] in [112, 157]:
                        raw_X = convert_to_158d(raw_X, y=np.array(self.recorded_labels))
                    if raw_X.shape[1] != config.SEQUENCE_LENGTH:
                        raw_X = resample_sequence_length(raw_X, config.SEQUENCE_LENGTH)
                self.recorded_sequences = list(raw_X)
                print(f"[*] Đã nạp sẵn {len(self.recorded_labels)} mẫu từ {self.dataset_file} (shape: {raw_X.shape})")
            except Exception as e:
                print(f"[!] Không thể đọc {self.dataset_file}: {e}")

        # Trạng thái ghi
        self.seq_len = config.SEQUENCE_LENGTH
        self.is_recording = False
        self.is_counting = False
        self.countdown_start = 0.0
        self.countdown_seconds = 3.0
        self.current_sequence: List[np.ndarray] = []
        self.current_frame_cache: List[np.ndarray] = []
        self.is_playback = False
        self.playback_idx = 0

    def get_class_counts(self) -> Dict[int, int]:
        """Đếm số mẫu hiện có của từng nhãn."""
        counts = {i: 0 for i in range(len(self.classes))}
        for lbl in self.recorded_labels:
            if lbl in counts:
                counts[lbl] += 1
        return counts

    def _draw_hud(self, frame: np.ndarray):
        """Vẽ bảng HUD điều khiển và thống kê trực quan."""
        h, w, _ = frame.shape

        # 1. Bảng điều khiển góc trên bên trái
        cv2.rectangle(frame, (10, 10), (420, 125), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (10, 10), (420, 125), (0, 200, 255), 1)

        cur_cls_name = self.classes[self.current_class_idx]
        counts = self.get_class_counts()
        cur_count = counts.get(self.current_class_idx, 0)

        cv2.putText(frame, f"MUC TIEU [{self.current_class_idx}]: {cur_cls_name}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Da co: {cur_count} mau | Tong cong: {len(self.recorded_labels)} mau", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        # Trạng thái hoạt động
        if self.is_recording:
            cv2.circle(frame, (25, 95), 8, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, f"DANG THU THAP ({len(self.current_sequence)}/{self.seq_len} frames)", (42, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        elif self.is_counting:
            cv2.putText(frame, "DANG DEM NGUOC CHUAN BI...", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif self.is_playback:
            cv2.putText(frame, f"PLAYBACK XEM LAI ({self.playback_idx+1}/{len(self.current_frame_cache)})", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "SAN SANG (Nhan 'R' de quay, '0-9' chon cu chi)", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

        # 2. Thanh phím tắt góc dưới
        cv2.rectangle(frame, (10, h - 35), (w - 10, h - 10), (15, 15, 15), cv2.FILLED)
        cv2.putText(frame, "Phim: [0-9] Chon | [R] Ghi | [SPACE] Playback | [Z] Xoa mau | [S] Luu file .npz | [Q] Thoat",
                    (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    def run(self):
        """Vòng lặp chính của giao diện thu thập dữ liệu."""
        if not self.cap.isOpened():
            print("[!] Không thể mở Webcam. Vui lòng kiểm tra lại kết nối camera.")
            return

        print("=" * 60)
        print("CÔNG CỤ THU THẬP CHUỖI CỬ CHỈ (INTERACTIVE SEQUENCE RECORDER)")
        print("  - Phím 0-9: Chọn cử chỉ mục tiêu")
        print("  - Phím 'R': Bắt đầu đếm ngược 3-2-1 và ghi 30 frame")
        print("  - Phím 'SPACE': Xem lại (Playback) chuỗi vừa quay")
        print("  - Phím 'Z': Xóa chuỗi vừa quay nếu sai động tác")
        print("  - Phím 'S': Lưu toàn bộ dữ liệu vào file .npz")
        print("  - Phím 'Q' hoặc ESC: Thoát")
        print("=" * 60)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Lật gương cho tự nhiên
                h, w, _ = frame.shape
                enhanced = self.enhancer.process(frame)
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

                # Phát hiện bàn tay với MediaPipe
                hands_data = []
                velocities = []
                if self.mp_hand_detector:
                    results = self.mp_hand_detector.process(rgb)
                    detected_count = len(results.multi_hand_landmarks) if (results and results.multi_hand_landmarks) else 0
                    for h_idx in range(min(2, detected_count)):
                        hand_lms = results.multi_hand_landmarks[h_idx]
                        mp_drawing.draw_landmarks(enhanced, hand_lms, mp_hands.HAND_CONNECTIONS)
                        raw_pts = [(i, l.x * w, l.y * h, l.z * w) for i, l in enumerate(hand_lms.landmark)]

                        # Lọc Kalman
                        smoothed_pts, is_valid = self.kalman_hands[h_idx].update(raw_pts)
                        if is_valid:
                            hands_data.append(smoothed_pts)
                            velocities.append(self.kalman_hands[h_idx].get_hand_velocity())

                    # Cho phép Kalman dead-reckoning khi tạm mất dấu
                    for h_idx in range(detected_count, 2):
                        smoothed_pts, is_valid = self.kalman_hands[h_idx].update(None)
                        if is_valid:
                            hands_data.append(smoothed_pts)
                            velocities.append(self.kalman_hands[h_idx].get_hand_velocity())

                # Xác định Mode ID cho cử chỉ mục tiêu (Mode 1: Chuột, Mode 2: Media/Cử chỉ động)
                if self.current_class_idx in [1, 2]:
                    target_mode = 1
                elif self.current_class_idx in [3, 4, 5, 6, 7, 8, 9]:
                    target_mode = 2
                else:
                    target_mode = 0

                # Trích xuất đặc trưng 158D Context Gate
                feat_158d = extract_158d_features(hands_data, velocities, mode_id=target_mode)

                # Xử lý đếm ngược 3-2-1
                if self.is_counting:
                    elapsed = time.time() - self.countdown_start
                    remaining = int(np.ceil(self.countdown_seconds - elapsed))
                    if remaining > 0:
                        cv2.putText(enhanced, f"CHUAN BI: {remaining}", (w // 2 - 150, h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 165, 255), 4)
                    else:
                        self.is_counting = False
                        self.is_recording = True
                        self.current_sequence = []
                        self.current_frame_cache = []

                # Chế độ ghi chuỗi SEQUENCE_LENGTH frame
                if self.is_recording:
                    self.current_sequence.append(feat_158d)
                    self.current_frame_cache.append(enhanced.copy())
                    if len(self.current_sequence) == self.seq_len:
                        self.recorded_sequences.append(np.array(self.current_sequence, dtype=np.float32))
                        self.recorded_labels.append(self.current_class_idx)
                        self.is_recording = False
                        self.is_playback = True
                        self.playback_idx = 0
                        print(f"[+] Đã ghi xong mẫu {len(self.recorded_sequences)} (Nhãn: {config.GESTURE_CLASSES[self.current_class_idx]}, Mode: {target_mode})")

                # Chế độ Playback
                if self.is_playback and len(self.current_frame_cache) > 0:
                    display_frame = self.current_frame_cache[self.playback_idx].copy()
                    self.playback_idx = (self.playback_idx + 1) % len(self.current_frame_cache)
                    cv2.putText(display_frame, "[CHE DO PLAYBACK XEM LAI]", (w - 320, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    display_frame = enhanced

                # Vẽ bảng HUD thông tin
                self._draw_hud(display_frame)

                cv2.imshow(f"HETHONGXULITAY - Sequence Recorder ({config.FEATURE_DIM}D)", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q') or key == 27:
                    break
                elif ord('0') <= key <= ord('9'):
                    self.current_class_idx = key - ord('0')
                elif key == ord('r') or key == ord('R'):
                    if not self.is_recording and not self.is_counting:
                        self.is_counting = True
                        self.countdown_start = time.time()
                        self.current_sequence = []
                        self.current_frame_cache = []
                        self.is_playback = False
                elif key == 32:  # Phím Space: Bật/Tắt Playback
                    if len(self.current_frame_cache) > 0:
                        self.is_playback = not self.is_playback
                elif key == ord('z') or key == ord('Z'):
                    if len(self.recorded_sequences) > 0:
                        self.recorded_sequences.pop()
                        self.recorded_labels.pop()
                        self.is_playback = False
                        self.current_frame_cache = []
                        print("[-] Đã xóa mẫu gần nhất vừa quay.")
                elif key == ord('s') or key == ord('S'):
                    if len(self.recorded_sequences) > 0:
                        X_arr = np.array(self.recorded_sequences, dtype=np.float32)
                        y_arr = np.array(self.recorded_labels, dtype=np.int64)
                        np.savez_compressed(self.dataset_file, X=X_arr, y=y_arr)
                        print(f"[+] Đã lưu an toàn {len(y_arr)} chuỗi vào {self.dataset_file}")

        finally:
            if self.cap.isOpened():
                self.cap.release()
            if self.mp_hand_detector is not None:
                try:
                    self.mp_hand_detector.close()
                except Exception:
                    pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    recorder = SequenceRecorder()
    recorder.run()
