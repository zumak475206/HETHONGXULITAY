"""
src/ui/workers.py
=================
Các luồng nền (QThread) cho Ứng dụng Desktop HETHONGXULITAY:
1. CameraWorker: Chạy camera 60 FPS, MediaPipe Hands, Dual-Pipeline (Mode 1 Chuột 0ms / Mode 2 Bi-LSTM 158D),
                 và nhận diện cử chỉ 2 tay mở để chuyển Mode tự động.
2. TrainWorker: Chạy tiến trình huấn luyện PyTorch trong nền, phát tín hiệu Epoch/Loss/Acc lên giao diện,
                xuất weights ONNX và kích hoạt Hot-Reload mô hình mà không làm đơ giao diện.
"""

import sys
import time
import queue
import collections
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

import config
from src.tracking.threaded_camera import ThreadedCamera
from src.preprocessing.clahe_enhancer import CLAHEEnhancer
from src.tracking.kalman_filter import HandKalmanFilter
from src.tracking.anatomical_guard import AnatomicalGuard
from src.modes.mode1_mouse import MouseController
from src.modes.mode2_presentation import PresentationController
from src.modes.mode3_sign_language import SignLanguageController
from src.features.feature_extractor import extract_158d_features
from src.inference.voting_smoother import VotingSmoother
from src.actions.dispatcher import ActionDispatcher

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    mp_hands = None
    mp_drawing = None


def draw_attention_heatmap(frame: np.ndarray, attention_weights: np.ndarray, x: int = 25, y: int = 420, w: int = 240, h: int = 20):
    """Vẽ thanh biểu đồ nhiệt Attention Heatmap hiển thị trọng số của chuỗi frame."""
    if attention_weights is None or len(attention_weights) == 0:
        return

    weights = np.array(attention_weights, dtype=np.float32)
    w_min, w_max = float(np.min(weights)), float(np.max(weights))
    norm_w = (weights - w_min) / max(1e-5, (w_max - w_min))

    cv2.rectangle(frame, (x - 2, y - 20), (x + w + 2, y + h + 2), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, f"ATTENTION HEATMAP ({len(norm_w)} FRAMES)", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    step = w / max(1, len(norm_w))
    for i, val in enumerate(norm_w):
        bx = int(x + i * step)
        bw = int(step) + 1
        color = cv2.applyColorMap(np.array([[[int(val * 255)]]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        col_bgr = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(frame, (bx, y), (bx + bw, y + h), col_bgr, cv2.FILLED)


class CameraWorker(QThread):
    """
    Luồng nền xử lý thị giác máy tính, bóc tách cử chỉ và điều khiển chuột.
    Giao tiếp với giao diện chính qua PyQt Signals an toàn và không gây lag.

    State Machine an toàn:
      STANDBY  --(2 tay mở 1.5s)--> MENU_OPEN
      MENU_OPEN --(1 ngón 1.5s)---> MODE1_MOUSE
      MENU_OPEN --(2 ngón 1.5s)---> MODE2_GESTURES
      MENU_OPEN --(3 ngón 1.5s)---> MODE3_SIGN
      MENU_OPEN --(4 ngón 1.5s)---> QUIT
      MODE_ANY  --(2 nắm 1.5s)----> MENU_OPEN
      MODE_ANY  --(1 nắm 1.5s)----> STANDBY (khoá màn hình)
    """

    frame_ready = pyqtSignal(np.ndarray)
    metrics_ready = pyqtSignal(float, float, str, str, str)  # fps, latency_ms, action, pipeline, guard
    mode_switched = pyqtSignal(str)                           # new_mode_name
    feature_ready = pyqtSignal(list)                          # 158D feature vector cho Studio thu thập mẫu
    subtitle_updated = pyqtSignal(dict)                       # Trạng thái phụ đề cho DesktopSubtitleOverlay
    error_occurred = pyqtSignal(str)
    show_mode_menu = pyqtSignal()                             # Yêu cầu UI hiển thị overlay menu chọn mode
    quit_app = pyqtSignal()                                   # Yêu cầu thoát toàn bộ ứng dụng

    # Hằng thời gian giữ cử chỉ (giây)
    GESTURE_HOLD_SEC: float = 1.5

    def __init__(self, initial_mode: str = "STANDBY", db=None, parent=None):
        super().__init__(parent)
        self.db = db
        if self.db is None:
            try:
                from src.database.db_manager import DatabaseManager
                self.db = DatabaseManager()
            except Exception:
                pass
        # App-level state machine: STANDBY | MENU_OPEN | MODE1_MOUSE | MODE2_GESTURES | MODE3_SIGN
        self.app_state: str = initial_mode
        self.current_mode = initial_mode   # backward-compat alias
        self.is_paused = False
        self.mouse_paused = False
        self.recording_mode_id: Optional[int] = None
        self._is_running = True

        # Trạng thái tham số động
        self.mouse_controller: Optional[MouseController] = None
        self.presentation_controller: Optional[PresentationController] = None
        self.sign_controller: Optional[SignLanguageController] = None
        self.anatomical_guard: Optional[AnatomicalGuard] = None
        self.enhancer: Optional[CLAHEEnhancer] = None
        self._new_ort_session: Optional[Any] = None
        self.two_hand_switch_time: float = 1.5  # kept for compat, state-machine uses GESTURE_HOLD_SEC

        # Hàng đợi điều phối hành động OS
        self.action_queue = queue.Queue(maxsize=15)
        self.dispatcher = ActionDispatcher(self.action_queue)

        # ── State-machine hold timer ──────────────────────────────────────────
        # gesture_hold_key: chuỗi mô tả cử chỉ đang được theo dõi
        # gesture_hold_start: thời điểm bắt đầu giữ cử chỉ (0 = chưa bắt đầu)
        self._gesture_hold_key: str = ""
        self._gesture_hold_start: float = 0.0
        self._gesture_cooldown: float = 0.0   # chống trigger liên tục

        # Cờ chuyển mode cũ (kept cho backward-compat, không dùng logic này nữa)
        self.open_hands_start_time = 0.0
        self.open_hands_cooldown = 0.0

        # Bộ đệm chuỗi thời gian
        self.temporal_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
        for _ in range(config.SEQUENCE_LENGTH):
            self.temporal_buffer.append([0.0] * config.FEATURE_DIM)

    # ── HELPER: đếm ngón tay duỗi ra ─────────────────────────────────────────

    @staticmethod
    def count_extended_fingers(hand_pts: list) -> int:
        """
        Trả về số ngón tay đang duỗi thẳng (0-5).
        hand_pts: list các tuple (idx, x, y, z) từ MediaPipe Hands (21 điểm).
        """
        if not hand_pts or len(hand_pts) < 21:
            return 0

        # Chuyển thành dict {idx: (x, y, z)}
        lm: Dict[int, tuple] = {p[0]: (p[1], p[2], p[3]) for p in hand_pts}

        # TIP và PIP indices cho 4 ngón (index → pinky)
        # Ngón duỗi khi TIP.y < PIP.y (y tăng xuống trong toạ độ chuẩn hoá)
        finger_pairs = [
            (8, 6),    # Index: TIP=8, PIP=6
            (12, 10),  # Middle: TIP=12, PIP=10
            (16, 14),  # Ring: TIP=16, PIP=14
            (20, 18),  # Pinky: TIP=20, PIP=18
        ]
        count = sum(
            1 for tip_i, pip_i in finger_pairs
            if lm[tip_i][1] < lm[pip_i][1]
        )

        # Ngón cái: so sánh theo trục x (cần biết bàn tay trái/phải)
        # Heuristic đơn giản: nếu wrist.x < middle_mcp.x → tay phải, ngược lại tay trái
        wrist_x = lm[0][0]
        mid_mcp_x = lm[9][0]
        if wrist_x < mid_mcp_x:  # bàn tay phải (nhìn từ camera)
            if lm[4][0] < lm[3][0]:
                count += 1
        else:                     # bàn tay trái
            if lm[4][0] > lm[3][0]:
                count += 1

        return min(5, count)

    @staticmethod
    def is_hand_open(hand_pts: list) -> bool:
        """Bàn tay xòe mở: >= 4 ngón duỗi."""
        return CameraWorker.count_extended_fingers(hand_pts) >= 4

    @staticmethod
    def is_hand_fisted(hand_pts: list) -> bool:
        """
        Bàn tay nắm lại thực sự (Clenched Fist): các ngón co chặt và không phải cử chỉ hình chữ O / Số 0.
        """
        if not hand_pts or len(hand_pts) < 21:
            return False
        if CameraWorker.count_extended_fingers(hand_pts) > 1:
            return False

        # Loại trừ hình chữ O / Số 0 (vòng tròn ngón cái chạm ngón trỏ):
        # Trong cử chỉ chữ O/Số 0, ngón trỏ vươn cong tạo vòng hở với ngón cái
        lm = {p[0]: np.array([p[1], p[2], p[3]], dtype=np.float32) for p in hand_pts}
        w = lm[0]
        middle_mcp = lm[9]
        palm_size = float(np.linalg.norm(middle_mcp - w))
        if palm_size < 1e-4:
            return False
        d_thumb_index = float(np.linalg.norm(lm[4] - lm[8])) / palm_size
        d_index_wrist = float(np.linalg.norm(lm[8] - w)) / palm_size

        # Nếu đầu ngón cái và đầu ngón trỏ áp sát tạo vòng tròn (< 0.50 palm_size) và vươn vòng (> 0.50 palm_size)
        # thì đây là ký hiệu số 0 / chữ O, không phải nắm đấm khóa màn hình
        if d_thumb_index < 0.50 and d_index_wrist > 0.50:
            return False

        return True

    def update_mouse_settings(
        self,
        sensitivity: Optional[float] = None,
        y_gain: Optional[float] = None,
        beta: Optional[float] = None,
        min_cutoff: Optional[float] = None,
        right_click_hold_time: Optional[float] = None,
        fist_grace_time: Optional[float] = None,
    ) -> None:
        """Cập nhật các tham số chuột động tới mouse_controller."""
        if self.mouse_controller:
            self.mouse_controller.update_settings(
                sensitivity=sensitivity,
                y_gain=y_gain,
                beta=beta,
                min_cutoff=min_cutoff,
                right_click_hold_time=right_click_hold_time,
                fist_grace_time=fist_grace_time,
            )

    def set_clahe_enabled(self, enabled: bool) -> None:
        """Bật hoặc tắt bộ lọc cải thiện ảnh CLAHE."""
        if self.enhancer:
            self.enhancer.enabled = bool(enabled)

    def set_anatomical_guard_enabled(self, enabled: bool) -> None:
        """Bật hoặc tắt bộ lọc bảo vệ sinh học Anatomical Guard."""
        if hasattr(self, "anatomical_guard") and self.anatomical_guard:
            self.anatomical_guard.enabled = bool(enabled)

    def reload_model(self, onnx_path: Optional[Any] = None) -> bool:
        """Hot-Reload mô hình mới nạp trực tiếp vào luồng camera đang chạy."""
        try:
            import onnxruntime as ort
            target_path = Path(onnx_path) if onnx_path else config.MODEL_ONNX_PATH
            if target_path.exists():
                new_session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
                self._new_ort_session = new_session
                if self.sign_controller:
                    self.sign_controller.reload_gesture_meanings()
                return True
        except Exception as e:
            print(f"[CameraWorker] Loi hot-reload model: {e}")
        return False

    def set_two_hand_switch_time(self, duration_sec: float) -> None:
        """Cập nhật thời gian giữ 2 tay để chuyển Mode."""
        self.two_hand_switch_time = max(0.5, float(duration_sec))

    def set_mode(self, mode_name: str) -> None:
        """Thay đổi chế độ hoạt động an toàn từ giao diện người dùng."""
        valid = ["STANDBY", "MODE1_MOUSE", "MODE2_GESTURES", "MODE3_SIGN"]
        if mode_name in valid:
            self.app_state = mode_name
            self.current_mode = mode_name
            # Reset buffer khi đổi chế độ
            self.temporal_buffer.clear()
            for _ in range(config.SEQUENCE_LENGTH):
                self.temporal_buffer.append([0.0] * config.FEATURE_DIM)
            self._reset_gesture_timer()

    def set_paused(self, paused: bool) -> None:
        """Bật / Tắt trạng thái tạm khóa (Standby Clutch)."""
        self.is_paused = paused
        if paused:
            self.app_state = "STANDBY"
            self.current_mode = "STANDBY"
        self._reset_gesture_timer()

    def _reset_gesture_timer(self) -> None:
        """Reset bộ đếm thời gian giữ cử chỉ."""
        self._gesture_hold_key = ""
        self._gesture_hold_start = 0.0

    def set_mouse_paused(self, paused: bool) -> None:
        """Tạm dừng điều khiển chuột OS (dành cho Studio, Training, Recording) mà không dừng Camera."""
        self.mouse_paused = paused
        if paused and self.mouse_controller:
            self.mouse_controller.cancel_click_and_drag()

    def set_recording_override(self, enabled: bool, mode_id: Optional[int] = None) -> None:
        """Kích hoạt chế độ quay mẫu: tự động ngắt chuột và gán đúng mode_id vào feature 158D."""
        self.mouse_paused = enabled
        self.recording_mode_id = mode_id if enabled else None
        if enabled and self.mouse_controller:
            self.mouse_controller.cancel_click_and_drag()

    def stop(self) -> None:
        """Dừng luồng an toàn và giải phóng tài nguyên."""
        self._is_running = False
        self.wait(1500)
        if hasattr(self, "dispatcher") and self.dispatcher:
            self.dispatcher.stop()

    def _update_gesture_state_machine(
        self,
        raw_hands_data: list,
        frame: np.ndarray,
        now: float,
    ) -> None:
        """
        State machine cử chỉ an toàn — gọi mỗi frame.
        Phát signal lên UI khi cần chuyển trạng thái.
        """
        h, w = frame.shape[:2]
        cx = w // 2

        # ── Xác định gesture hiện tại ─────────────────────────────────────────
        n_hands = len(raw_hands_data)
        gesture_key = ""

        if self.app_state == "STANDBY":
            # Kích hoạt menu: 2 tay xòe mở
            if n_hands == 2:
                open0 = self.is_hand_open(raw_hands_data[0])
                open1 = self.is_hand_open(raw_hands_data[1])
                if open0 and open1:
                    gesture_key = "BOTH_OPEN"

        elif self.app_state == "MENU_OPEN":
            # Chọn mode bằng số ngón tay (1/2/3/4) — chỉ cần 1 tay
            if n_hands >= 1:
                # Lấy số ngón của bàn tay có ngón nhiều nhất để ổn định hơn
                fingers = max(self.count_extended_fingers(h_pts) for h_pts in raw_hands_data)
                if fingers in (1, 2, 3, 4):
                    gesture_key = f"FINGERS_{fingers}"

        else:
            # Đang ở một mode nào đó (MODE1/2/3)
            if n_hands == 2:
                fist0 = self.is_hand_fisted(raw_hands_data[0])
                fist1 = self.is_hand_fisted(raw_hands_data[1])
                if fist0 and fist1:
                    gesture_key = "BOTH_FIST"  # → về menu
            elif n_hands == 1 and self.current_mode != "MODE3_SIGN":
                # Trong Mode 3 (Ký hiệu), người dùng thường xuyên dùng 1 tay làm ký hiệu (số 0, nắm, chữ O, v.v.)
                # nên cấm dùng 1 tay để khóa màn hình; chỉ Mode 1 & 2 mới dùng 1 nắm tay để khóa
                if self.is_hand_fisted(raw_hands_data[0]):
                    gesture_key = "ONE_FIST"   # → khóa màn hình / standby

        # ── Bộ đếm thời gian giữ ─────────────────────────────────────────────
        if gesture_key:
            if gesture_key != self._gesture_hold_key:
                # Cử chỉ mới — reset timer
                self._gesture_hold_key = gesture_key
                self._gesture_hold_start = now
            else:
                duration = now - self._gesture_hold_start
                progress = min(1.0, duration / self.GESTURE_HOLD_SEC)
                self._draw_hold_progress(frame, cx, progress, gesture_key)

                if duration >= self.GESTURE_HOLD_SEC:
                    self._trigger_gesture(gesture_key, now)
        else:
            # Không có cử chỉ khớp — reset
            self._gesture_hold_key = ""
            self._gesture_hold_start = 0.0

    def _draw_hold_progress(
        self,
        frame: np.ndarray,
        cx: int,
        progress: float,
        key: str,
    ) -> None:
        """Vẽ vòng cung tiến trình trên frame để người dùng biết cử chỉ đang được nhận."""
        h = frame.shape[0]
        center = (cx, 55)
        radius = 32
        angle = int(360 * progress)
        bar_color = {
            "BOTH_OPEN": (0, 220, 255),   # cyan  → mở menu
            "FINGERS_1": (16, 185, 129),  # green → mode 1
            "FINGERS_2": (251, 191, 36),  # amber → mode 2
            "FINGERS_3": (99, 102, 241),  # indigo → mode 3
            "FINGERS_4": (244, 63, 94),   # rose  → thoát
            "BOTH_FIST": (251, 191, 36),  # amber → về menu
            "ONE_FIST":  (244, 63, 94),   # rose  → khóa
        }.get(key, (200, 200, 200))

        label_map = {
            "BOTH_OPEN": "MO MENU...",
            "FINGERS_1": "MODE 1 CHUOT...",
            "FINGERS_2": "MODE 2 SLIDES...",
            "FINGERS_3": "MODE 3 KY HIEU...",
            "FINGERS_4": "THOAT APP...",
            "BOTH_FIST": "VE MENU...",
            "ONE_FIST":  "KHOA MAN HINH...",
        }
        label = label_map.get(key, key)

        # Nền tối cho vòng cung
        cv2.circle(frame, center, radius + 4, (12, 18, 35), cv2.FILLED)
        # Vòng nền
        cv2.circle(frame, center, radius, (40, 50, 70), 4)
        # Vòng cung tiến trình
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle,
                    bar_color, 4, cv2.LINE_AA)
        # Phần trăm bên trong
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(frame, pct_text, (center[0] - 16, center[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Nhãn bên dưới vòng
        cv2.putText(frame, label, (cx - len(label) * 4, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, bar_color, 1)

    def _trigger_gesture(self, gesture_key: str, now: float) -> None:
        """Thực thi chuyển trạng thái sau khi giữ đủ thời gian."""
        self._gesture_hold_key = ""
        self._gesture_hold_start = 0.0
        self._gesture_cooldown = now + 2.0  # cooldown 2s

        if gesture_key == "BOTH_OPEN" and self.app_state == "STANDBY":
            self.app_state = "MENU_OPEN"
            self.current_mode = "STANDBY"
            self.show_mode_menu.emit()

        elif gesture_key == "FINGERS_1" and self.app_state == "MENU_OPEN":
            self._enter_mode("MODE1_MOUSE")

        elif gesture_key == "FINGERS_2" and self.app_state == "MENU_OPEN":
            self._enter_mode("MODE2_GESTURES")

        elif gesture_key == "FINGERS_3" and self.app_state == "MENU_OPEN":
            self._enter_mode("MODE3_SIGN")

        elif gesture_key == "FINGERS_4" and self.app_state == "MENU_OPEN":
            self.quit_app.emit()

        elif gesture_key == "BOTH_FIST" and self.app_state not in ("STANDBY", "MENU_OPEN"):
            self.app_state = "MENU_OPEN"
            self.current_mode = "STANDBY"
            self.show_mode_menu.emit()

        elif gesture_key == "ONE_FIST" and self.app_state not in ("STANDBY", "MENU_OPEN"):
            self.app_state = "STANDBY"
            self.current_mode = "STANDBY"
            self.is_paused = False          # camera vẫn chạy nhưng mode tắt
            self.mode_switched.emit("STANDBY")

    def _enter_mode(self, mode_name: str) -> None:
        """Chuyển vào mode cụ thể từ menu."""
        self.app_state = mode_name
        self.current_mode = mode_name
        self.is_paused = False
        self.temporal_buffer.clear()
        for _ in range(config.SEQUENCE_LENGTH):
            self.temporal_buffer.append([0.0] * config.FEATURE_DIM)
        self.mode_switched.emit(mode_name)

    def run(self) -> None:
        """Vòng lặp chính xử lý thị giác máy tính 60 FPS."""
        # 1. Khởi chạy Action Dispatcher
        if not self.dispatcher.is_alive():
            self.dispatcher.start()

        # 2. Khởi tạo Camera đa luồng Single-Latch
        camera = None
        try:
            camera = ThreadedCamera(
                src=config.CAMERA_INDEX,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS,
                buffersize=getattr(config, "CAMERA_BUFFERSIZE", 1),
            )
        except Exception as e:
            self.error_occurred.emit(f"Lỗi khởi tạo Camera: {e}")

        # 3. Khởi tạo các module thị giác
        enhancer = CLAHEEnhancer(
            enabled=config.CLAHE_ENABLED,
            clip_limit=config.CLAHE_CLIP_LIMIT,
            grid_size=config.CLAHE_GRID_SIZE,
            auto_gamma=config.CLAHE_AUTO_GAMMA,
            anti_glare=config.CLAHE_ANTI_GLARE,
            edge_sharpen=getattr(config, "CLAHE_EDGE_SHARPEN", True),
        )
        self.enhancer = enhancer
        kalman_hands = [
            HandKalmanFilter(
                grace_frames=config.KALMAN_GRACE_FRAMES,
                dt=1.0 / float(config.CAMERA_FPS),
                process_noise_var=config.KALMAN_PROCESS_NOISE_VAR,
                measurement_noise_var=config.KALMAN_MEASUREMENT_NOISE_VAR,
            ),
            HandKalmanFilter(
                grace_frames=config.KALMAN_GRACE_FRAMES,
                dt=1.0 / float(config.CAMERA_FPS),
                process_noise_var=config.KALMAN_PROCESS_NOISE_VAR,
                measurement_noise_var=config.KALMAN_MEASUREMENT_NOISE_VAR,
            ),
        ]
        smoother = VotingSmoother(
            window_size=config.VOTING_WINDOW_SIZE,
            min_ratio=config.VOTING_MIN_RATIO,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            cooldown_sec=config.ACTION_COOLDOWN_SEC,
        )
        anatomical_guard = AnatomicalGuard(
            enabled=getattr(config, "ANATOMICAL_GUARD_ENABLED", False),
            max_finger_palm_ratio=getattr(config, "MAX_FINGER_PALM_RATIO", 1.60),
            max_segment_palm_ratio=getattr(config, "MAX_SEGMENT_PALM_RATIO", 1.10),
            min_phalanx_ratio=getattr(config, "MIN_PHALANX_RATIO", 0.15),
            max_phalanx_ratio=getattr(config, "MAX_PHALANX_RATIO", 6.00),
        )
        self.anatomical_guard = anatomical_guard
        mouse_controller = MouseController(
            frame_w=config.CAMERA_WIDTH,
            frame_h=config.CAMERA_HEIGHT,
        )
        self.mouse_controller = mouse_controller

        presentation_controller = PresentationController(
            frame_w=config.CAMERA_WIDTH,
            frame_h=config.CAMERA_HEIGHT,
            action_queue=self.action_queue,
        )
        self.presentation_controller = presentation_controller

        sign_controller = SignLanguageController(
            frame_w=config.CAMERA_WIDTH,
            frame_h=config.CAMERA_HEIGHT,
            db=getattr(self, "db", None),
            action_queue=self.action_queue,
        )
        self.sign_controller = sign_controller

        mp_detector = None
        if mp_hands:
            try:
                mp_detector = mp_hands.Hands(
                    max_num_hands=config.MAX_NUM_HANDS,
                    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                )
            except Exception:
                mp_detector = None

        # 4. Nạp mô hình Deep Learning ONNX (hoặc fallback PyTorch)
        ort_session = None
        torch_model = None
        engine_name = "Chưa nạp"
        if config.MODEL_ONNX_PATH.exists():
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(str(config.MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
                inp_dim = sess.get_inputs()[0].shape[-1]
                if isinstance(inp_dim, int) and inp_dim == config.FEATURE_DIM:
                    ort_session = sess
                    engine_name = "FP32 ONNX"
            except Exception:
                pass

        fps = 30.0
        prev_time = time.time()
        curr_stable_label = "IDLE"
        curr_confidence = 0.0
        attn_weights = np.zeros(config.SEQUENCE_LENGTH, dtype=np.float32)

        # Vòng lặp chính
        while self._is_running:
            loop_start = time.time()

            # Hot-Reload mô hình mới nếu vừa huấn luyện xong
            if self._new_ort_session is not None:
                ort_session = self._new_ort_session
                self._new_ort_session = None

            frame = None
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue
            else:
                # Nếu không có camera thật, sinh frame giả lập để kiểm thử và giao diện hoạt động bình thường
                frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "KHONG TIM THAY CAMERA (CHE DO GIA LAP - SIMULATION)", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"Che do hien tai: {self.current_mode}", (60, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
                time.sleep(0.033)

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Nếu đang ở chế độ Tạm Khóa (Standby Clutch)
            if self.is_paused:
                cv2.rectangle(frame, (0, 0), (w, h), (10, 15, 30), cv2.FILLED)
                cv2.putText(frame, "TAM DUNG / STANDBY CLUTCH", (w // 2 - 200, h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                cv2.putText(frame, "Click nut 'MO KHOA' ben duoi de tiep tuc", (w // 2 - 180, h // 2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                self.frame_ready.emit(frame)
                self.metrics_ready.emit(0.0, 0.0, "Tam dung (Standby)", "Tat", "Nghi")
                time.sleep(0.04)
                continue

            # Xử lý CLAHE nâng cấp chất lượng ảnh
            enhanced = enhancer.process(frame)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            hands_data = []
            raw_hands_data = []
            velocities = []
            guard_status = "BẬT (An toàn)" if (self.anatomical_guard and self.anatomical_guard.enabled) else "TẮT (Mượt 100% MediaPipe)"

            if mp_detector:
                results = mp_detector.process(rgb)
                detected_lms = results.multi_hand_landmarks if (results and results.multi_hand_landmarks) else []
                detected_count = len(detected_lms)

                # Thuật toán ghép cặp vị trí chống hoán đổi tay
                if detected_count == 2 and kalman_hands[0].has_active_hand and kalman_hands[1].has_active_hand:
                    prev_w0 = kalman_hands[0].filters[0].x[0:2, 0]
                    prev_w1 = kalman_hands[1].filters[0].x[0:2, 0]
                    cur_w0 = np.array([detected_lms[0].landmark[0].x * w, detected_lms[0].landmark[0].y * h])
                    cur_w1 = np.array([detected_lms[1].landmark[0].x * w, detected_lms[1].landmark[0].y * h])
                    if np.linalg.norm(cur_w1 - prev_w0) + np.linalg.norm(cur_w0 - prev_w1) < np.linalg.norm(cur_w0 - prev_w0) + np.linalg.norm(cur_w1 - prev_w1):
                        detected_lms = [detected_lms[1], detected_lms[0]]

                for h_idx in range(min(2, detected_count)):
                    hand_lms = detected_lms[h_idx]
                    if hand_lms is not None:
                        raw_pts = [(i, l.x * w, l.y * h, l.z * w) for i, l in enumerate(hand_lms.landmark)]

                        # Bộ lọc giải phẫu học sinh học (Anatomical Guard)
                        if anatomical_guard.enabled:
                            is_valid_anatomy, guard_reason = anatomical_guard.validate_hand(raw_pts)
                            if not is_valid_anatomy:
                                guard_status = f"CHẶN: {guard_reason}"
                                cv2.putText(enhanced, f"CHAN DI DANG: {guard_reason}", (15, h - 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                smoothed_pts, is_valid = kalman_hands[h_idx].update(None)
                                if is_valid:
                                    hands_data.append(smoothed_pts)
                                    velocities.append(kalman_hands[h_idx].get_hand_velocity())
                                continue
                            else:
                                guard_status = "BẬT (An toàn)"
                        else:
                            guard_status = "TẮT (Mượt 100% MediaPipe)"

                        # Vẽ khung xương bàn tay
                        if mp_drawing:
                            mp_drawing.draw_landmarks(enhanced, hand_lms, mp_hands.HAND_CONNECTIONS)

                        raw_hands_data.append(raw_pts)

                        smoothed_pts, is_valid = kalman_hands[h_idx].update(raw_pts)
                        if is_valid:
                            hands_data.append(smoothed_pts)
                            velocities.append(kalman_hands[h_idx].get_hand_velocity())
                    else:
                        smoothed_pts, is_valid = kalman_hands[h_idx].update(None)
                        if is_valid:
                            hands_data.append(smoothed_pts)
                            velocities.append(kalman_hands[h_idx].get_hand_velocity())

                for h_idx in range(detected_count, 2):
                    smoothed_pts, is_valid = kalman_hands[h_idx].update(None)
                    if is_valid:
                        hands_data.append(smoothed_pts)
                        velocities.append(kalman_hands[h_idx].get_hand_velocity())

            # ── STATE MACHINE CỬ CHỈ AN TOÀN ────────────────────────────────
            now = time.time()
            if not getattr(self, "mouse_paused", False) and now > self._gesture_cooldown:
                self._update_gesture_state_machine(raw_hands_data, enhanced, now)

            # ── DUAL-PIPELINE: XỬ LÝ THEO CHẾ ĐỘ HIỆN TẠI ──
            action_text = "Cho cu chi"
            pipeline_text = "Dual-Pipeline"

            # Luôn trích đặc trưng 158D (phục vụ Studio/recording)
            rec_mode = getattr(self, "recording_mode_id", None)
            if rec_mode is not None:
                mode_val = int(rec_mode)
            else:
                mode_val = config.MODE_MAP.get(self.current_mode, 1 if self.current_mode == "MODE1_MOUSE" else 2)
            feat_158d = extract_158d_features(hands_data, velocities, mode_id=mode_val)
            self.temporal_buffer.append(feat_158d)
            self.feature_ready.emit(feat_158d)

            # Khi STANDBY hoặc MENU_OPEN: hiển thị overlay hướng dẫn, không chạy mode nào
            if self.app_state in ("STANDBY", "MENU_OPEN"):
                if self.app_state == "STANDBY":
                    pipeline_text = "STANDBY - Cho cu chi"
                    action_text = "XOE 2 TAY 1.5s de mo Menu"
                    # Vẽ overlay STANDBY trên frame
                    overlay = enhanced.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (5, 10, 22), -1)
                    cv2.addWeighted(overlay, 0.55, enhanced, 0.45, 0, enhanced)
                    cv2.putText(enhanced, "STANDBY - CHO CU CHI", (w // 2 - 160, h // 2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(enhanced, "XOE 2 TAY 1.5s -> MENU CHON MODE",
                                (w // 2 - 215, h // 2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                else:
                    pipeline_text = "MENU - Chon Mode"
                    action_text = "Giu so ngon: 1=Mode1 2=Mode2 3=Mode3 4=Thoat"
                    # Vẽ overlay MENU trên frame
                    overlay = enhanced.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (5, 10, 22), -1)
                    cv2.addWeighted(overlay, 0.50, enhanced, 0.50, 0, enhanced)
                    cv2.putText(enhanced, "MENU CHON MODE", (w // 2 - 110, h // 2 - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
                    items = [
                        (" 1 ngon → MODE 1: CHUOT", (16, 185, 129)),
                        (" 2 ngon → MODE 2: SLIDES", (251, 191, 36)),
                        (" 3 ngon → MODE 3: KY HIEU", (99, 102, 241)),
                        (" 4 ngon → THOAT APP", (244, 63, 94)),
                    ]
                    for i, (txt, col) in enumerate(items):
                        cv2.putText(enhanced, txt, (w // 2 - 160, h // 2 - 20 + i * 36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 1, cv2.LINE_AA)
                    cv2.putText(enhanced, "Giu cu chi 1.5s de xac nhan",
                                (w // 2 - 165, h // 2 + 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

            elif self.current_mode == "MODE1_MOUSE":
                if not getattr(self, "mouse_paused", False):
                    # Mode 1: Chuột tức thì Zero-Lag (0ms)
                    pipeline_text = "Zero-Lag (Raw + OneEuro)"
                    active_hand = []
                    if len(raw_hands_data) == 1:
                        active_hand = raw_hands_data[0]
                    elif len(raw_hands_data) > 1:
                        # Ưu tiên bàn tay đang duỗi ngón trỏ
                        hand_scores = []
                        for h_pts in raw_hands_data:
                            f_up = mouse_controller.count_fingers_up(h_pts)
                            score = 2 if f_up[1] == 1 else (1 if sum(f_up) > 0 else 0)
                            hand_scores.append(score)
                        active_hand = raw_hands_data[int(np.argmax(hand_scores))]

                    mouse_controller.process(enhanced, active_hand)
                    action_text = mouse_controller.action_text or "Di chuot (0ms)"
                else:
                    pipeline_text = "Studio / Recording (Chuột tạm dừng)"
                    action_text = "Tạm dừng chuột (Chế độ Studio/Quay)"

                self.subtitle_updated.emit({
                    "sentence": "",
                    "current_word": "",
                    "current_meaning": "",
                    "confidence": 0.0,
                    "is_active": False,
                })

            elif self.current_mode == "MODE2_GESTURES":
                # Mode 2: Trình chiếu Slide (Kinematic Fast-Swipe + L-Sign F5)
                pipeline_text = "Mode 2: Trinh Chieu Slide"
                if not getattr(self, "mouse_paused", False):
                    presentation_controller.process(enhanced, raw_hands_data)
                    action_text = presentation_controller.action_text or "San sang trinh chieu"
                else:
                    pipeline_text = "Studio / Recording (Mode 2 tam dung)"
                    action_text = "Tam dung Mode 2 (Che do Studio/Quay)"

                self.subtitle_updated.emit({
                    "sentence": "",
                    "current_word": "",
                    "current_meaning": "",
                    "confidence": 0.0,
                    "is_active": False,
                })

            elif self.current_mode == "MODE3_SIGN":
                # Mode 3: Nhận diện Ngôn ngữ ký hiệu VSL & Phụ đề máy tính (158D Context Gate)
                pipeline_text = "Mode 3: VSL Sign Language (158D)"
                has_hands = (len(raw_hands_data) > 0)
                num_classes = len(config.GESTURE_CLASSES)
                logits = np.zeros(num_classes, dtype=np.float32)
                attn_weights = np.zeros(config.SEQUENCE_LENGTH, dtype=np.float32)

                if len(self.temporal_buffer) == config.SEQUENCE_LENGTH and has_hands:
                    seq_np = np.array([self.temporal_buffer], dtype=np.float32)
                    if ort_session is not None:
                        try:
                            ort_outs = ort_session.run(None, {"input_sequence": seq_np})
                            logits = ort_outs[0][0]
                            if len(ort_outs) > 1 and ort_outs[1] is not None:
                                attn_weights = ort_outs[1][0]
                        except Exception:
                            pass

                if self.sign_controller is not None:
                    raw_lbl, conf, sentence = self.sign_controller.process_inference(
                        logits=logits,
                        attn_weights=attn_weights,
                        class_list=config.GESTURE_CLASSES,
                        has_hands=has_hands,
                        raw_hands_data=raw_hands_data,
                    )
                    self.sign_controller.draw_hud(enhanced, hands_data=hands_data, raw_hands_data=raw_hands_data)
                    sub_state = self.sign_controller.get_subtitle_state()
                    self.subtitle_updated.emit(sub_state)

                    cur_meaning = sub_state.get("current_meaning", "") or sub_state.get("current_word", "Chờ ký hiệu")
                    action_text = f"VSL: {cur_meaning} ({sub_state.get('confidence', 0.0)*100:.0f}%)"
                else:
                    action_text = "VSL: Sẵn sàng dịch"
            else:
                # Trạng thái STANDBY hoặc MENU_OPEN
                pipeline_text = "Standby"
                action_text = "Chờ cử chỉ"
                self.subtitle_updated.emit({
                    "sentence": "",
                    "current_word": "",
                    "current_meaning": "",
                    "confidence": 0.0,
                    "is_active": False,
                })

            # Tính toán FPS và độ trễ
            loop_end = time.time()
            latency_ms = (loop_end - loop_start) * 1000.0
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (loop_end - prev_time)))
            prev_time = loop_end

            # Phát tín hiệu lên giao diện
            self.frame_ready.emit(enhanced)
            self.metrics_ready.emit(fps, latency_ms, action_text, pipeline_text, guard_status)

        # Giải phóng camera khi thoát luồng
        if camera:
            camera.release()


class TrainWorker(QThread):
    """
    Luồng huấn luyện mô hình Bi-LSTM nơ-ron nền (Background Training Worker).
    Huấn luyện không nghẽn GUI, phát tín hiệu Epoch tiến trình và Hot-Reload.
    """

    epoch_progress = pyqtSignal(int, int, float, float)  # epoch, total_epochs, loss, acc
    training_completed = pyqtSignal(dict)                # summary stats
    training_failed = pyqtSignal(str)

    def __init__(
        self,
        db=None,
        epochs: int = 35,
        batch_size: int = 32,
        sequences_dir: Optional[Any] = None,
        include_legacy: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.db = db
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequences_dir = sequences_dir
        self.include_legacy = include_legacy
        self._is_cancelled = False

    def cancel(self) -> None:
        """Hủy quá trình huấn luyện an toàn."""
        self._is_cancelled = True

    def run(self) -> None:
        """Thực thi tiến trình huấn luyện mô hình PyTorch thực tế."""
        try:
            from src.database.dataset_merger import load_merged_dataset
            from src.models.bilstm_attention import MultiScaleBiLSTMAttention
            from src.training.train import export_to_onnx, quantize_onnx_int8

            X, y, classes = load_merged_dataset(
                db=self.db,
                sequences_dir=self.sequences_dir,
                include_legacy=self.include_legacy,
                filter_empty_hands=True,
            )
            if len(X) == 0:
                self.training_failed.emit("Không tìm thấy mẫu nào trong cơ sở dữ liệu để huấn luyện!")
                return

            # Thêm mẫu cơ sở cho Mode 3 (IDLE, OPEN_PALM với Context Gate = 0.75)
            # Giúp mô hình hiểu rằng trong Mode 3, tay nghỉ vẫn là IDLE, tay xòe vẫn là OPEN_PALM!
            extra_m3_X = []
            extra_m3_y = []
            for base_name in ["IDLE", "OPEN_PALM", "FIST_LOCK"]:
                if base_name in classes:
                    b_idx = classes.index(base_name)
                    b_indices = [i for i, lbl in enumerate(y) if lbl == b_idx]
                    for idx in b_indices[:20]:
                        seq_copy = np.array(X[idx], copy=True)
                        seq_copy[:, 157] = 3.0 / 4.0  # Mode 3 Gate = 0.75
                        extra_m3_X.append(seq_copy)
                        extra_m3_y.append(b_idx)
            if extra_m3_X:
                X = np.concatenate([X, np.array(extra_m3_X, dtype=np.float32)], axis=0)
                y = np.concatenate([y, np.array(extra_m3_y, dtype=np.int64)], axis=0)

            # Tăng cường dữ liệu (Geometric Data Augmentation & Oversampling) cho các lớp thiểu số
            target_samples = max(40, int(np.median(np.bincount(y))))
            class_counts = np.bincount(y, minlength=len(classes))

            X_aug_list = list(X)
            y_aug_list = list(y)

            for c_idx in range(len(classes)):
                cnt = class_counts[c_idx]
                if 0 < cnt < target_samples:
                    needed = target_samples - cnt
                    c_indices = [i for i, label in enumerate(y) if label == c_idx]
                    for k in range(needed):
                        src_idx = c_indices[k % len(c_indices)]
                        src_seq = np.array(X[src_idx], copy=True)
                        # Biến thiên hình học: Scale tọa độ 0.96 - 1.04
                        scale = np.random.uniform(0.96, 1.04)
                        src_seq[:, 0:63] *= scale
                        src_seq[:, 76:139] *= scale

                        # Nhiễu góc khớp ngón tay
                        angle_jitter = np.random.normal(0, 0.015, size=src_seq[:, 63:73].shape).astype(np.float32)
                        src_seq[:, 63:73] = np.clip(src_seq[:, 63:73] + angle_jitter, 0.0, 1.0)

                        # Context Gate Dropout (25% cơ hội set gate=0.0 để mạng học đặc trưng hình học ngón tay)
                        if np.random.rand() < 0.25:
                            src_seq[:, 157] = 0.0

                        X_aug_list.append(src_seq)
                        y_aug_list.append(c_idx)

            X_aug_arr = np.array(X_aug_list, dtype=np.float32)
            y_aug_arr = np.array(y_aug_list, dtype=np.int64)

            total_epochs = self.epochs
            batch_sz = max(1, min(self.batch_size, len(X_aug_arr)))

            # Dataset & DataLoader PyTorch
            X_tensor = torch.tensor(X_aug_arr, dtype=torch.float32)
            y_tensor = torch.tensor(y_aug_arr, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=True)

            device = torch.device("cuda:0" if (torch.cuda.is_available() and "cuda" in config.DEVICE) else "cpu")

            num_classes = max(2, len(classes))
            model = MultiScaleBiLSTMAttention(
                input_dim=config.FEATURE_DIM,
                cnn_branch_channels=32,
                lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
                lstm_num_layers=config.LSTM_NUM_LAYERS,
                lstm_dropout=config.LSTM_DROPOUT,
                num_classes=num_classes,
            ).to(device)

            # Nạp trọng số Backbone tiền huấn luyện nếu có (Warm-Start Transfer Learning)
            if config.MODEL_PT_PATH.exists():
                try:
                    state = torch.load(config.MODEL_PT_PATH, map_location=device)
                    model_state = model.state_dict()
                    compatible_state = {
                        k: v for k, v in state.items()
                        if k in model_state and v.shape == model_state[k].shape
                    }
                    model_state.update(compatible_state)
                    model.load_state_dict(model_state)
                except Exception:
                    pass

            # Trọng số phân lớp cân bằng
            aug_counts = np.bincount(y_aug_arr, minlength=len(classes))
            weights = len(y_aug_arr) / (len(classes) * np.maximum(1, aug_counts))
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.03)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

            final_loss = 1.0
            final_acc = 0.0

            for ep in range(1, total_epochs + 1):
                if self._is_cancelled:
                    self.training_failed.emit("Người dùng đã hủy huấn luyện.")
                    return

                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()

                    logits, _ = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * len(batch_y)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += len(batch_y)

                scheduler.step()
                epoch_loss = running_loss / max(1, total)
                epoch_acc = (correct / max(1, total)) * 100.0
                final_loss = epoch_loss
                final_acc = epoch_acc

                self.epoch_progress.emit(ep, total_epochs, float(epoch_loss), float(epoch_acc))

            # Lưu weights PyTorch
            config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(model.state_dict(), config.MODEL_PT_PATH)
            except Exception as e:
                print(f"[TrainWorker] Warning: Không thể lưu weights .pt ({e})")

            # Cập nhật danh sách class toàn hệ thống & ghi file persistent
            config.GESTURE_CLASSES = list(classes)
            config.NUM_CLASSES = len(classes)
            try:
                import json
                with open(config.CLASSES_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(list(classes), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # Xuất sang ONNX FP32
            export_to_onnx(model, config.MODEL_ONNX_PATH, input_dim=config.FEATURE_DIM, seq_len=config.SEQUENCE_LENGTH)

            # Lượng tử hóa INT8
            try:
                quantize_onnx_int8(config.MODEL_ONNX_PATH, config.MODEL_INT8_PATH, model_cpu=model.to("cpu"))
            except Exception:
                pass

            summary = {
                "total_samples": len(X),
                "total_classes": len(classes),
                "final_loss": round(final_loss, 4),
                "final_acc": round(final_acc, 2),
                "model_path": str(config.MODEL_ONNX_PATH.name),
            }
            self.training_completed.emit(summary)

        except Exception as e:
            self.training_failed.emit(f"Lỗi huấn luyện: {str(e)}")
