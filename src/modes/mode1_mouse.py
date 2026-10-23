"""
src/modes/mode1_mouse.py
========================
Chế độ 1: Điều khiển chuột máy tính và hệ thống bằng cử chỉ tay (Phiên bản siêu mượt 2.0).
Kế thừa trọn vẹn tinh hoa công thái học từ HeThongXuLiBangTay:
1. Di chuột tức thì (Zero-Lag 0ms latency): Tọa độ ngón trỏ lọc qua One Euro Filter thích ứng vận tốc.
2. Pinch Position Lock: Khóa cứng con trỏ chuột khi chạm ngón cái vào ngón trỏ để không bị lệch mục tiêu.
3. Máy trạng thái Kéo-Thả (Drag & Drop): Nhấp nhả <0.5s -> Left Click; giữ >=0.5s hoặc rê >45px -> mouseDown (Kéo thả), tách ngón -> mouseUp.
4. Click Phải Chữ V (V-Sign Hold 0.35s): Xòe 2 ngón với vòng tròn tiến trình Radial Progress Ring.
5. Cần gạt Cuộn trang (Anchor Joystick Scroll): Khép 2 ngón tạo tâm neo, kéo lệch lên/xuống cuộn liên tục với gia tốc đàn hồi.
6. Vùng điều khiển thu gọn (Margin Box 20-25%) & Gia tốc trục Y phi tuyến (1.55x).
7. Nắm đấm nghỉ tay (Fist Lock): Tạm dừng di chuột, hủy mọi click dở dang, có grace period 0.25s khi mở tay.
8. Chỉnh Âm lượng (Shaka) / Độ sáng (Rock-on) ngầm qua AsyncSystemController (background worker thread 25Hz).
"""

import time
import math
import collections
import threading
from typing import List, Tuple, Optional, Any, Union
import numpy as np
import cv2
import pyautogui

import config
from src.tracking.one_euro_filter import PointFilter

# Cấu hình PyAutoGUI phản hồi thời gian thực
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# ── Hỗ trợ Âm lượng Windows qua pycaw ──────────────────────────────────────────
try:
    from pycaw.pycaw import AudioUtilities
    _speakers = AudioUtilities.GetSpeakers()
    _audio_volume = _speakers.EndpointVolume
except Exception:
    _audio_volume = None

# ── Hỗ trợ Độ sáng qua screen_brightness_control ──────────────────────────────
try:
    import screen_brightness_control as sbc
    _has_sbc = True
except Exception:
    _has_sbc = False


# ── Tiện ích vẽ HUD ────────────────────────────────────────────────────────────

def put_text_outlined(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_thickness: int = 4,
) -> np.ndarray:
    """Vẽ chữ có viền đen để tương phản sắc nét trên mọi nền ảnh camera."""
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
        outline_color, outline_thickness, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA
    )
    return frame


def draw_rect_transparent(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int] = (30, 30, 30),
    alpha: float = 0.55,
) -> np.ndarray:
    """Vẽ hình chữ nhật bán trong suốt bằng cv2.addWeighted."""
    h, w = frame.shape[:2]
    x1_c, x2_c = max(0, min(w, x1)), max(0, min(w, x2))
    y1_c, y2_c = max(0, min(h, y1)), max(0, min(h, y2))
    if x2_c <= x1_c or y2_c <= y1_c:
        return frame

    roi = frame[y1_c:y2_c, x1_c:x2_c]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2_c - x1_c, y2_c - y1_c), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1_c:y2_c, x1_c:x2_c] = roi
    return frame


# ── Hệ thống điều khiển Âm lượng & Độ sáng nền ────────────────────────────────

class AsyncSystemController:
    """
    Thực thi lệnh thay đổi Volume và Brightness ở background worker thread.
    Hoàn toàn KHÔNG làm nghẽn OpenCV rendering loop (duy trì 30-60 FPS).
    """

    def __init__(self):
        self._target_vol: Optional[float] = None
        self._target_bright: Optional[int] = None
        self._lock = threading.Lock()
        self._worker_running = True

        self.cached_vol_pct = 50
        self.cached_bright_pct = 50
        if _audio_volume is not None:
            try:
                self.cached_vol_pct = int(_audio_volume.GetMasterVolumeLevelScalar() * 100)
            except Exception:
                pass
        if _has_sbc:
            try:
                b = sbc.get_brightness()
                if b and len(b) > 0:
                    self.cached_bright_pct = int(b[0])
            except Exception:
                pass

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def set_volume(self, scalar: float) -> None:
        scalar = max(0.0, min(1.0, float(scalar)))
        with self._lock:
            self._target_vol = scalar
            self.cached_vol_pct = int(scalar * 100)

    def set_brightness(self, pct: int) -> None:
        pct = max(0, min(100, int(pct)))
        with self._lock:
            self._target_bright = pct
            self.cached_bright_pct = pct

    def get_vol_pct(self) -> int:
        return self.cached_vol_pct

    def get_bright_pct(self) -> int:
        return self.cached_bright_pct

    def stop(self) -> None:
        self._worker_running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.3)
            except Exception:
                pass

    def _run_loop(self) -> None:
        last_applied_vol = None
        last_applied_bright = None
        while self._worker_running:
            vol_to_set = None
            bright_to_set = None
            with self._lock:
                if self._target_vol is not None and self._target_vol != last_applied_vol:
                    vol_to_set = self._target_vol
                    last_applied_vol = vol_to_set
                if self._target_bright is not None and self._target_bright != last_applied_bright:
                    bright_to_set = self._target_bright
                    last_applied_bright = bright_to_set

            if vol_to_set is not None and _audio_volume is not None:
                try:
                    _audio_volume.SetMasterVolumeLevelScalar(vol_to_set, None)
                except Exception:
                    pass

            if bright_to_set is not None and _has_sbc:
                try:
                    sbc.set_brightness(bright_to_set)
                except Exception:
                    pass

            time.sleep(0.04)  # 25Hz worker loop


# ── Bộ lọc làm mượt trạng thái ngón tay (Finger Smoother) ─────────────────────

class FingerSmoother:
    """
    Bộ lọc đa số 5-frame để loại bỏ hiện tượng nhảy trạng thái ngón tay (debouncing).
    """

    def __init__(self, window: int = 5, threshold: float = 0.6):
        self._window = window
        self._threshold = threshold
        self._history = [collections.deque(maxlen=window) for _ in range(5)]
        self._initialized = False

    def update(self, raw_fingers: List[int]) -> List[int]:
        if not self._initialized:
            for i, val in enumerate(raw_fingers):
                self._history[i].clear()
                for _ in range(self._window):
                    self._history[i].append(val)
            self._initialized = True
            return list(raw_fingers)

        result = []
        for i, val in enumerate(raw_fingers):
            self._history[i].append(val)
            ratio = sum(self._history[i]) / float(self._window)
            result.append(1 if ratio >= self._threshold else 0)
        return result

    def reset(self) -> None:
        for d in self._history:
            d.clear()
        self._initialized = False


# ── Bộ điều khiển chuột chính (MouseController) ───────────────────────────────

class MouseController:
    """
    Bộ điều khiển chuột siêu mượt cho Mode 1, tích hợp đầy đủ công thái học vi mô.
    """

    def __init__(
        self,
        frame_w: int = getattr(config, "CAMERA_WIDTH", 640),
        frame_h: int = getattr(config, "CAMERA_HEIGHT", 480),
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Kích thước màn hình
        try:
            self.screen_w, self.screen_h = pyautogui.size()
        except Exception:
            self.screen_w, self.screen_h = 1920, 1080

        # Margin box thu gọn về tâm
        margin_x_ratio = getattr(config, "CONTROL_REGION_MARGIN_X", 0.25)
        margin_y_ratio = getattr(config, "CONTROL_REGION_MARGIN_Y", 0.20)
        self.margin_x = int(frame_w * margin_x_ratio)
        self.margin_y = int(frame_h * margin_y_ratio)
        self.sensitivity = float(getattr(config, "MOUSE_SENSITIVITY", 1.25))

        # Bộ lọc tọa độ
        self.prev_mouse_x = float(self.screen_w // 2)
        self.prev_mouse_y = float(self.screen_h // 2)

        # One Euro Filter
        min_cutoff = float(getattr(config, "MOUSE_ONE_EURO_MIN_CUTOFF", 1.0))
        beta = float(getattr(config, "MOUSE_ONE_EURO_BETA", 0.008))
        self.cursor_filter = PointFilter(min_cutoff=min_cutoff, beta=beta)
        self.y_gain = float(getattr(config, "MOUSE_Y_DOWN_GAIN", 1.55))
        self.right_click_hold_time = float(getattr(config, "RIGHT_CLICK_HOLD_TIME", 0.35))
        self.fist_grace_time = float(getattr(config, "FIST_GRACE_TIME", 0.25))

        # Deadzone
        self.deadzone = float(getattr(config, "MOUSE_DEADZONE", 0.0))

        # Pinch Position Lock
        self.locked_x: Optional[float] = None
        self.locked_y: Optional[float] = None

        # Trạng thái Click trái & Kéo thả
        self.is_mouse_down = False
        self.down_start_time: Optional[float] = None
        self.down_pos = (0.0, 0.0)
        self.down_cam_pos = (0.0, 0.0)
        self.is_dragging = False
        self.last_left_click_time = 0.0
        self.left_click_cooldown = 0.20

        # Trạng thái Cuộn trang
        self.scroll_anchor_x: Optional[float] = None
        self.scroll_anchor_y: Optional[float] = None
        self.prev_scroll_y: Optional[float] = None
        self.scroll_accumulator = 0.0
        self.last_scroll_time = 0.0

        # Trạng thái Click phải (Chữ V Hold 0.35s)
        self.v_sign_hold_start: Optional[float] = None
        self.v_sign_clicked = False
        self.v_sign_frames = 0
        self.last_right_click_time = 0.0
        self.right_click_cooldown = float(getattr(config, "RIGHT_CLICK_COOLDOWN", 1.0))

        # Fist Lock & Grace Period
        self.last_fist_time = 0.0
        self._was_fist = False
        self.just_unfisted = False

        # Hệ thống ngầm
        self.sys_ctrl = AsyncSystemController()

        # Quản lý Mode (MOUSE / VOLUME / BRIGHTNESS)
        self.current_mode = "MOUSE"
        self.mode_hold_start: Optional[float] = None
        self.mode_exit_start: Optional[float] = None
        self.target_switching_mode: Optional[str] = None
        self.MODE_SWITCH_HOLD_TIME = 0.35

        # Finger smoother nội bộ
        self.finger_smoother = FingerSmoother(window=5, threshold=0.6)

        # Thông báo HUD
        self.action_text = ""
        self.action_color = (255, 255, 255)
        self.action_time = 0.0

    def set_frame_dimensions(self, frame_w: int, frame_h: int) -> None:
        """Cập nhật kích thước khung hình camera và tính lại margin box động."""
        if frame_w <= 0 or frame_h <= 0:
            return
        if self.frame_w == frame_w and self.frame_h == frame_h:
            return
        self.frame_w = frame_w
        self.frame_h = frame_h
        margin_x_ratio = getattr(config, "CONTROL_REGION_MARGIN_X", 0.25)
        margin_y_ratio = getattr(config, "CONTROL_REGION_MARGIN_Y", 0.20)
        self.margin_x = int(frame_w * margin_x_ratio)
        self.margin_y = int(frame_h * margin_y_ratio)

    def update_settings(
        self,
        sensitivity: Optional[float] = None,
        y_gain: Optional[float] = None,
        beta: Optional[float] = None,
        min_cutoff: Optional[float] = None,
        right_click_hold_time: Optional[float] = None,
        fist_grace_time: Optional[float] = None,
    ) -> None:
        """Cập nhật động các tham số điều khiển chuột tại runtime mà không cần khởi động lại."""
        if sensitivity is not None:
            self.sensitivity = max(0.1, float(sensitivity))
        if y_gain is not None:
            self.y_gain = max(0.5, float(y_gain))
        if right_click_hold_time is not None:
            self.right_click_hold_time = max(0.05, float(right_click_hold_time))
        if fist_grace_time is not None:
            self.fist_grace_time = max(0.05, float(fist_grace_time))
        if beta is not None and hasattr(self, "cursor_filter"):
            b = float(beta)
            self.cursor_filter.x_filter.beta = b
            self.cursor_filter.y_filter.beta = b
        if min_cutoff is not None and hasattr(self, "cursor_filter"):
            mc = float(min_cutoff)
            self.cursor_filter.x_filter.min_cutoff = mc
            self.cursor_filter.y_filter.min_cutoff = mc

    # ── Tiện ích hình học nội bộ ──────────────────────────────────────────────

    @staticmethod
    def _get_pt(lm_list: Any, idx: int) -> Tuple[float, float]:
        """Lấy (x, y) bất kể format [(id, x, y, z)], [(id, x, y)] hay [(x, y)]."""
        pt = lm_list[idx]
        if hasattr(pt, "x") and hasattr(pt, "y"):
            return float(pt.x), float(pt.y)
        if isinstance(pt, dict):
            return float(pt["x"]), float(pt["y"])
        if len(pt) >= 3:
            if isinstance(pt[0], (int, np.integer)) and int(pt[0]) == idx:
                return float(pt[1]), float(pt[2])
            return float(pt[0]), float(pt[1])
        if len(pt) == 2:
            return float(pt[0]), float(pt[1])
        return 0.0, 0.0

    def _dist(self, lm_list: Any, id_a: int, id_b: int) -> float:
        x1, y1 = self._get_pt(lm_list, id_a)
        x2, y2 = self._get_pt(lm_list, id_b)
        return math.hypot(x2 - x1, y2 - y1)

    def get_palm_size(self, lm_list: Any) -> float:
        """Kích thước bàn tay bất biến (Invariant Palm Size)."""
        if not lm_list or len(lm_list) < 21:
            return 1.0
        w_mid = self._dist(lm_list, config.WRIST, config.MIDDLE_MCP)
        knuckle = self._dist(lm_list, config.INDEX_MCP, config.PINKY_MCP)
        return max(w_mid, knuckle * 1.25, 1.0)

    def _thumb_angle(self, lm_list: Any) -> float:
        """Góc tại khớp THUMB_IP (3)."""
        ax, ay = self._get_pt(lm_list, config.THUMB_MCP)
        bx, by = self._get_pt(lm_list, config.THUMB_IP)
        cx, cy = self._get_pt(lm_list, config.THUMB_TIP)
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag = math.hypot(*v1) * math.hypot(*v2) + 1e-6
        cos_val = max(-1.0, min(1.0, dot / mag))
        return math.degrees(math.acos(cos_val))

    def _is_thumb_extended(self, lm_list: Any) -> bool:
        """Kiểm tra ngón cái duỗi theo hình học 3D bất biến."""
        angle = self._thumb_angle(lm_list)
        if angle < 140.0:
            return False

        dist_tip_pinky = self._dist(lm_list, config.THUMB_TIP, config.PINKY_MCP)
        dist_ip_pinky  = self._dist(lm_list, config.THUMB_IP,  config.PINKY_MCP)
        palm_size = self.get_palm_size(lm_list)
        dist_tip_index_mcp = self._dist(lm_list, config.THUMB_TIP, config.INDEX_MCP)

        if dist_tip_index_mcp < palm_size * 0.22:
            return False

        return dist_tip_pinky > dist_ip_pinky * 1.02

    def count_fingers_up(self, lm_list: Any) -> List[int]:
        """Đếm trạng thái 5 ngón [cái, trỏ, giữa, áp út, út] (1 = duỗi, 0 = gập)."""
        if not lm_list or len(lm_list) < 21:
            return [0, 0, 0, 0, 0]

        fingers = [1 if self._is_thumb_extended(lm_list) else 0]
        tips = [config.INDEX_TIP, config.MIDDLE_TIP, config.RING_TIP, config.PINKY_TIP]
        pips = [config.INDEX_PIP, config.MIDDLE_PIP, config.RING_PIP, config.PINKY_PIP]
        mcps = [config.INDEX_MCP, config.MIDDLE_MCP, config.RING_MCP, config.PINKY_MCP]

        for tip_id, pip_id, mcp_id in zip(tips, pips, mcps):
            dist_tip_w = self._dist(lm_list, tip_id, config.WRIST)
            dist_pip_w = self._dist(lm_list, pip_id, config.WRIST)
            dist_tip_m = self._dist(lm_list, tip_id, mcp_id)
            dist_pip_m = self._dist(lm_list, pip_id, mcp_id)
            is_extended = (dist_tip_w > dist_pip_w) and (dist_tip_m > dist_pip_m)
            fingers.append(1 if is_extended else 0)

        return fingers

    # ── State Machine & Actions ────────────────────────────────────────────────

    def set_action(self, text: str, color: Tuple[int, int, int] = (0, 255, 255)) -> None:
        self.action_text = text
        self.action_color = color
        self.action_time = time.time()

    def cancel_click_and_drag(self) -> None:
        """Hủy bỏ ngay lập tức tiến trình Click/Hold/Drag đang chờ, nhả chuột an toàn."""
        if self.is_mouse_down or self.is_dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        self.is_mouse_down = False
        self.is_dragging = False
        self.down_start_time = None
        self.down_pos = (0.0, 0.0)
        self.down_cam_pos = (0.0, 0.0)
        self.locked_x = None
        self.locked_y = None

    def reset(self) -> None:
        """Reset toàn bộ trạng thái điều khiển và bộ lọc One Euro Filter."""
        self.cursor_filter.reset()
        self.just_unfisted = True
        self.last_fist_time = time.time()  # Grace buffer 0.25s chống giật và click nhầm khi reset/unpause
        self._was_fist = False
        self.locked_x = None
        self.locked_y = None
        self.down_start_time = None
        self.scroll_anchor_x = None
        self.scroll_anchor_y = None
        self.prev_scroll_y = None
        self.scroll_accumulator = 0.0
        self.last_scroll_time = 0.0
        self.v_sign_hold_start = None
        self.v_sign_clicked = False
        self.v_sign_frames = 0
        self.current_mode = "MOUSE"
        self.mode_hold_start = None
        self.mode_exit_start = None
        self.target_switching_mode = None
        self.cancel_click_and_drag()
        self.finger_smoother.reset()

    def process(
        self,
        frame: np.ndarray,
        lm_list: Any,
        tracker: Any = None,
        smoother: Any = None,
    ) -> None:
        """
        Xử lý cử chỉ chuột và hệ thống mỗi frame với Zero-Lag (0ms latency).
        """
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            fh, fw = frame.shape[:2]
            if fw != self.frame_w or fh != self.frame_h:
                self.set_frame_dimensions(fw, fh)

        self._draw_control_box(frame)

        if not lm_list or len(lm_list) < 21:
            self.cancel_click_and_drag()
            self.locked_x = None
            self.locked_y = None
            self.down_start_time = None
            self.scroll_anchor_x = None
            self.scroll_anchor_y = None
            self.prev_scroll_y = None
            self.scroll_accumulator = 0.0
            self.last_scroll_time = 0.0
            self.v_sign_hold_start = None
            self.v_sign_clicked = False
            self.v_sign_frames = 0
            self.current_mode = "MOUSE"
            self.cursor_filter.reset()
            self.just_unfisted = True
            self._draw_action_hud(frame)
            return

        curr_time = time.time()

        # 1. Kích thước lòng bàn tay bất biến
        palm_size = self.get_palm_size(lm_list)

        # 2. Đếm trạng thái ngón tay
        if tracker is not None and hasattr(tracker, "count_fingers_up"):
            raw_fingers = tracker.count_fingers_up(lm_list)
        else:
            raw_fingers = self.count_fingers_up(lm_list)

        if smoother is not None and hasattr(smoother, "update"):
            fingers = smoother.update(raw_fingers)
        else:
            fingers = self.finger_smoother.update(raw_fingers)

        # Tọa độ các đầu ngón
        tx, ty = self._get_pt(lm_list, config.THUMB_TIP)
        ix, iy = self._get_pt(lm_list, config.INDEX_TIP)
        mx, my = self._get_pt(lm_list, config.MIDDLE_TIP)

        # ── 3. KIỂM TRA NẮM TAY (FIST LOCK) ──────────────────────────────────
        is_fist = (
            (fingers[1:] == [0, 0, 0, 0])
            or (sum(fingers) <= 1 and fingers[1] == 0)
            or (raw_fingers[1:] == [0, 0, 0, 0])
        )

        if is_fist:
            self.last_fist_time = curr_time
            self.just_unfisted = True
            self.cursor_filter.reset()
            self.cancel_click_and_drag()
            self.scroll_anchor_x = None
            self.scroll_anchor_y = None
            self.prev_scroll_y = None
            self.scroll_accumulator = 0.0
            self.last_scroll_time = 0.0
            self.v_sign_hold_start = None
            self.v_sign_clicked = False
            self.v_sign_frames = 0
            self.mode_hold_start = None
            self.set_action("NAM TAY (Khoa chuot)", (120, 120, 120))
            self._draw_action_hud(frame)
            return

        # Grace period khi vừa mở nắm tay ra
        is_unfisting_grace = (curr_time - self.last_fist_time < getattr(self, "fist_grace_time", 0.25))

        # ── 3b. KIỂM TRA CỬ CHỈ ÂM LƯỢNG (SHAKA) & ĐỘ SÁNG (ROCK-ON) ───────────
        # Shaka (Âm lượng): Cái + Út duỗi, Trỏ + Giữa + Áp út gập
        is_shaka = (
            fingers[0] == 1 and fingers[4] == 1
            and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0
        ) and not is_unfisting_grace

        # Rock-on (Độ sáng): Trỏ + Út duỗi, Giữa + Áp út gập
        is_rock_on = (
            fingers[1] == 1 and fingers[4] == 1
            and fingers[2] == 0 and fingers[3] == 0
        ) and not is_unfisting_grace

        # Quản lý chuyển đổi chế độ
        if self.current_mode == "MOUSE":
            if is_shaka or is_rock_on:
                candidate = "BRIGHTNESS" if is_rock_on else "VOLUME"
                if self.mode_hold_start is None:
                    self.mode_hold_start = curr_time
                    self.target_switching_mode = candidate
                elif self.target_switching_mode == candidate:
                    elapsed = curr_time - self.mode_hold_start
                    progress = min(1.0, elapsed / self.MODE_SWITCH_HOLD_TIME)

                    cx, cy = (int(ix), int(iy)) if is_rock_on else (int(tx), int(ty))
                    mode_lbl = "DO SANG" if is_rock_on else "AM LUONG"
                    mode_col = (0, 215, 255) if is_rock_on else (0, 220, 255)

                    cv2.circle(frame, (cx, cy), 22, (50, 50, 50), 2)
                    cv2.ellipse(frame, (cx, cy), (22, 22), -90, 0, int(progress * 360), mode_col, 3)
                    put_text_outlined(frame, f"Giu {mode_lbl}...", (cx - 45, cy - 30),
                                      scale=0.45, color=mode_col, thickness=1, outline_thickness=2)

                    if elapsed >= self.MODE_SWITCH_HOLD_TIME:
                        self.current_mode = self.target_switching_mode
                        self.mode_hold_start = None
                        self.mode_exit_start = None
                        self.set_action(f"MO CHE DO: {self.current_mode}", mode_col)
                else:
                    self.mode_hold_start = curr_time
                    self.target_switching_mode = candidate
            else:
                self.mode_hold_start = None
        else:
            in_valid_gesture = (
                (self.current_mode == "VOLUME" and is_shaka)
                or (self.current_mode == "BRIGHTNESS" and is_rock_on)
            )
            if not in_valid_gesture:
                if self.mode_exit_start is None:
                    self.mode_exit_start = curr_time
                elif curr_time - self.mode_exit_start >= 0.25:
                    self.current_mode = "MOUSE"
                    self.mode_exit_start = None
                    self.just_unfisted = True
                    self.cursor_filter.reset()
            else:
                self.mode_exit_start = None

        # ── 4. XỬ LÝ CHỈNH ÂM LƯỢNG (SHAKA) ───────────────────────────────────
        if self.current_mode == "VOLUME":
            mmx, mmy = self._get_pt(lm_list, config.MIDDLE_MCP)
            vol_pct = int(np.interp(mmy, [90, 390], [100, 0]))
            vol_pct = max(0, min(100, vol_pct))
            self.sys_ctrl.set_volume(vol_pct / 100.0)
            self.set_action(f"AM LUONG: {vol_pct}%", (0, 220, 255))
            self._draw_slider(frame, "VOL", vol_pct, (0, 220, 255), x_pos=30)
            cv2.line(frame, (50, int(mmy)), (int(mmx), int(mmy)), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (int(mmx), int(mmy)), 8, (0, 220, 255), cv2.FILLED)
            self._draw_action_hud(frame)
            return

        # ── 5. XỬ LÝ CHỈNH ĐỘ SÁNG (ROCK-ON) ───────────────────────────────────
        if self.current_mode == "BRIGHTNESS":
            dist_ti = self._dist(lm_list, config.THUMB_TIP, config.INDEX_TIP)
            ratio = dist_ti / palm_size
            bright_pct = int(np.interp(ratio, [0.28, 1.05], [0, 100]))
            bright_pct = max(0, min(100, bright_pct))
            self.sys_ctrl.set_brightness(bright_pct)
            self.set_action(f"DO SANG: {bright_pct}%", (0, 215, 255))
            self._draw_slider(frame, "BRIGHT", bright_pct, (0, 215, 255), x_pos=30)
            mid_x, mid_y = int((tx + ix) / 2), int((ty + iy) / 2)
            cv2.line(frame, (int(tx), int(ty)), (int(ix), int(iy)), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (mid_x, mid_y), 8, (0, 180, 255), cv2.FILLED)
            self._draw_action_hud(frame)
            return

        # ── 6. CỬ CHỈ 2 NGÓN: CUỘN TRANG (ANCHOR JOYSTICK) HOẶC CLICK PHẢI (CHỮ V)
        is_two_fingers = (
            fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0
        ) and not is_unfisting_grace

        if is_two_fingers:
            self.cancel_click_and_drag()
            dist_im = self._dist(lm_list, config.INDEX_TIP, config.MIDDLE_TIP)
            im_ratio = dist_im / palm_size
            scroll_gap_max = getattr(config, "SCROLL_FINGER_GAP_MAX", 0.28)
            v_gap_min = getattr(config, "RIGHT_CLICK_V_GAP_MIN", 0.35)
            mid_x, mid_y = (ix + mx) / 2.0, (iy + my) / 2.0

            # ── 6a. CUỘN TRANG (2 NGÓN KHÉP SÁT) ──────────────────────────────
            if im_ratio < scroll_gap_max:
                self.v_sign_hold_start = None
                self.v_sign_clicked = False
                self.locked_x = None
                self.locked_y = None

                scroll_mode = getattr(config, "SCROLL_MODE", "ANCHOR")
                if scroll_mode == "ANCHOR":
                    if self.scroll_anchor_x is None or self.scroll_anchor_y is None:
                        self.scroll_anchor_x = mid_x
                        self.scroll_anchor_y = mid_y
                        self.last_scroll_time = curr_time
                        self.scroll_accumulator = 0.0

                    dt = curr_time - self.last_scroll_time
                    self.last_scroll_time = curr_time
                    dt = max(0.001, min(0.1, dt))

                    anchor_x = self.scroll_anchor_x
                    anchor_y = self.scroll_anchor_y
                    offset_y = mid_y - anchor_y
                    norm_offset = offset_y / palm_size

                    deadzone = float(getattr(config, "SCROLL_ANCHOR_DEADZONE_RATIO", 0.04))
                    max_offset = float(getattr(config, "SCROLL_ANCHOR_MAX_OFFSET_RATIO", 0.35))
                    max_speed = float(getattr(config, "SCROLL_ANCHOR_MAX_SPEED", 72.0))
                    min_speed = float(getattr(config, "SCROLL_ANCHOR_MIN_SPEED", 6.0))

                    if abs(norm_offset) <= deadzone:
                        scroll_label = "DUNG YEN (O)"
                        status_col = (0, 255, 255)
                        self.scroll_accumulator = 0.0
                    else:
                        progress = min(1.0, (abs(norm_offset) - deadzone) / max(1e-5, (max_offset - deadzone)))
                        speed = min_speed + (max_speed - min_speed) * (progress ** 1.3)
                        direction = 1 if norm_offset > 0 else -1

                        self.scroll_accumulator += direction * speed * dt
                        clicks = int(self.scroll_accumulator)
                        if clicks != 0:
                            try:
                                pyautogui.scroll(-clicks)
                            except Exception:
                                pass
                            self.scroll_accumulator -= float(clicks)

                        pct = int(progress * 100)
                        if direction > 0:
                            scroll_label = f"CUON XUONG {pct}% ({speed:.1f}/s)"
                            status_col = (0, 200, 255)
                            self.set_action(f"CUON XUONG ({pct}%)", status_col)
                        else:
                            scroll_label = f"CUON LEN {pct}% ({speed:.1f}/s)"
                            status_col = (0, 255, 180)
                            self.set_action(f"CUON LEN ({pct}%)", status_col)

                    # Vẽ HUD Anchor Joystick
                    deadzone_px = max(6, int(palm_size * deadzone))
                    cv2.circle(frame, (int(anchor_x), int(anchor_y)), deadzone_px, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (int(anchor_x), int(anchor_y)), 4, (0, 255, 255), cv2.FILLED)
                    cv2.line(frame, (int(anchor_x), int(anchor_y)), (int(mid_x), int(mid_y)), status_col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(mid_x), int(mid_y)), 8, status_col, cv2.FILLED)
                    put_text_outlined(frame, f"JOYSTICK: {scroll_label}", (int(mid_x) - 75, int(mid_y) - 20),
                                      scale=0.48, color=status_col, thickness=1, outline_thickness=3)
                    self._draw_action_hud(frame)
                    return
                else:
                    # Fallback Touchpad mode
                    self.scroll_anchor_x = None
                    self.scroll_anchor_y = None
                    scroll_sensitivity = float(getattr(config, "SCROLL_SENSITIVITY", 15.0))
                    deadzone_ratio = float(getattr(config, "SCROLL_DEADZONE_RATIO", 0.008))
                    if self.prev_scroll_y is not None:
                        delta_y = mid_y - self.prev_scroll_y
                        norm_delta = delta_y / palm_size
                        if abs(norm_delta) > deadzone_ratio:
                            self.scroll_accumulator += norm_delta * scroll_sensitivity
                            clicks = int(self.scroll_accumulator)
                            if clicks != 0:
                                try:
                                    pyautogui.scroll(-clicks)
                                except Exception:
                                    pass
                                self.scroll_accumulator -= float(clicks)
                    self.prev_scroll_y = mid_y
                    self._draw_action_hud(frame)
                    return

            # ── 6b. CLICK PHẢI (CHỮ V: 2 NGÓN XÒE RỘNG) ────────────────────────
            elif im_ratio >= v_gap_min:
                self.scroll_anchor_x = None
                self.scroll_anchor_y = None
                self.prev_scroll_y = None
                self.scroll_accumulator = 0.0

                if self.locked_x is None:
                    self.locked_x = self.prev_mouse_x
                    self.locked_y = self.prev_mouse_y

                hold_time_req = float(getattr(self, "right_click_hold_time", getattr(config, "RIGHT_CLICK_HOLD_TIME", 0.35)))
                if self.v_sign_clicked:
                    put_text_outlined(frame, "DA CLICK PHAI", (int(mid_x) - 55, int(mid_y) - 28),
                                      scale=0.48, color=(0, 215, 255), thickness=1, outline_thickness=3)
                    cv2.circle(frame, (int(ix), int(iy)), 10, (0, 140, 255), 2)
                    cv2.circle(frame, (int(mx), int(my)), 10, (0, 140, 255), 2)
                    self._draw_action_hud(frame)
                    return

                if self.v_sign_hold_start is None:
                    self.v_sign_hold_start = curr_time

                hold_duration = curr_time - self.v_sign_hold_start
                progress = min(1.0, max(0.0, hold_duration / hold_time_req))

                if hold_duration >= hold_time_req and (curr_time - self.last_right_click_time > self.right_click_cooldown):
                    try:
                        pyautogui.rightClick()
                    except Exception:
                        pass
                    self.last_right_click_time = curr_time
                    self.v_sign_clicked = True
                    self.v_sign_hold_start = None
                    self.set_action("DA CLICK PHAI", (0, 140, 255))
                    self._draw_action_hud(frame)
                    return

                # Radial Progress Ring
                cv2.circle(frame, (int(mid_x), int(mid_y)), 20, (60, 60, 60), 2)
                cv2.ellipse(frame, (int(mid_x), int(mid_y)), (20, 20), -90, 0, int(progress * 360), (0, 140, 255), 3)
                put_text_outlined(frame, f"CLICK PHAI (V) {int(progress * 100)}%", (int(mid_x) - 70, int(mid_y) - 28),
                                  scale=0.45, color=(0, 140, 255), thickness=1, outline_thickness=2)
                self._draw_action_hud(frame)
                return
            else:
                self.scroll_anchor_x = None
                self.scroll_anchor_y = None
                self.prev_scroll_y = None
                self.scroll_accumulator = 0.0
                self.v_sign_hold_start = None
                self.v_sign_clicked = False
                self.locked_x = None
                self.locked_y = None
                self._draw_action_hud(frame)
                return
        else:
            self.scroll_anchor_x = None
            self.scroll_anchor_y = None
            self.prev_scroll_y = None
            self.scroll_accumulator = 0.0
            self.v_sign_hold_start = None
            self.v_sign_clicked = False
            self.locked_x = None
            self.locked_y = None

        # ── 7. DI CHUỘT LIÊN TỤC & LÀM MƯỢT (ONE EURO FILTER) ────────────────
        if self.current_mode == "MOUSE":
            box_x1 = self.margin_x
            box_x2 = self.frame_w - self.margin_x
            box_y1 = self.margin_y
            box_y2 = self.frame_h - self.margin_y
            center_x = (box_x1 + box_x2) / 2.0
            center_y = (box_y1 + box_y2) / 2.0

            dx = (ix - center_x) * self.sensitivity

            # Gia tốc phi tuyến trục Y khi gập cổ tay
            dy_raw = iy - center_y
            if dy_raw > 0:
                max_down_range = max(1.0, box_y2 - center_y)
                down_ratio = min(1.0, max(0.0, dy_raw / max_down_range))
                down_gain_extra = max(0.0, float(getattr(self, "y_gain", getattr(config, "MOUSE_Y_DOWN_GAIN", 1.55))) - 1.0)
                gain_y = 1.0 + down_gain_extra * (down_ratio ** 1.5)
                dy = dy_raw * self.sensitivity * gain_y
            else:
                dy = dy_raw * self.sensitivity

            adj_ix = center_x + dx
            adj_iy = center_y + dy

            raw_target_x = np.interp(adj_ix, [box_x1, box_x2], [0, self.screen_w])
            raw_target_y = np.interp(adj_iy, [box_y1, box_y2], [0, self.screen_h])

            if self.just_unfisted:
                self.cursor_filter.reset()
                self.prev_mouse_x = float(raw_target_x)
                self.prev_mouse_y = float(raw_target_y)
                self.just_unfisted = False

            filtered_x, filtered_y = self.cursor_filter.filter(raw_target_x, raw_target_y, curr_time)
            dist_moved = math.hypot(filtered_x - self.prev_mouse_x, filtered_y - self.prev_mouse_y)

            if is_unfisting_grace:
                self.down_start_time = None
                self.locked_x = None
                self.locked_y = None
                calc_x = max(2, min(self.screen_w - 2, int(filtered_x)))
                calc_y = max(2, min(self.screen_h - 2, int(filtered_y)))
                if calc_x != int(self.prev_mouse_x) or calc_y != int(self.prev_mouse_y):
                    try:
                        pyautogui.moveTo(calc_x, calc_y)
                    except Exception:
                        pass
                    self.prev_mouse_x = float(calc_x)
                    self.prev_mouse_y = float(calc_y)
                cv2.circle(frame, (int(ix), int(iy)), 8, (0, 255, 0), cv2.FILLED)
                self._draw_action_hud(frame)
                return

            # ── 8. CLICK TRÁI & KÉO THẢ (DRAG & DROP) ──────────────────────────
            ipx, ipy = self._get_pt(lm_list, config.INDEX_PIP)
            idx, idy = self._get_pt(lm_list, config.INDEX_DIP)

            dist_pip = math.hypot(tx - ipx, ty - ipy)
            dist_dip = math.hypot(tx - idx, ty - idy)
            dist_tip = math.hypot(tx - ix, ty - iy)
            touch_dist = min(dist_pip, dist_dip, dist_tip)
            touch_ratio = touch_dist / palm_size

            TOUCH_ENTER    = float(getattr(config, "MOUSE_TOUCH_ENTER", 0.28))
            TOUCH_RELEASE  = float(getattr(config, "MOUSE_TOUCH_RELEASE", 0.32))
            DROP_RELEASE   = float(getattr(config, "MOUSE_DROP_RELEASE", 0.38))
            DRAG_HOLD_TIME = float(getattr(config, "MOUSE_DRAG_HOLD_TIME", 0.50))
            TOUCH_LOCK     = float(getattr(config, "MOUSE_TOUCH_LOCK", 0.35))

            if dist_pip <= dist_dip and dist_pip <= dist_tip:
                target_pt = (ipx, ipy)
            elif dist_dip <= dist_pip and dist_dip <= dist_tip:
                target_pt = (idx, idy)
            else:
                target_pt = (ix, iy)
            mid_x = int((tx + target_pt[0]) / 2)
            mid_y = int((ty + target_pt[1]) / 2)

            if touch_ratio < TOUCH_ENTER:
                if self.down_start_time is None:
                    self.down_start_time = curr_time
                    self.down_pos = (filtered_x, filtered_y)
                    self.down_cam_pos = (ix, iy)

                held_duration = curr_time - self.down_start_time
                dist_from_down = math.hypot(filtered_x - self.down_pos[0], filtered_y - self.down_pos[1])
                dist_cam_from_down = math.hypot(ix - self.down_cam_pos[0], iy - self.down_cam_pos[1])

                drag_dist_ratio = float(getattr(config, "MOUSE_DRAG_DISTANCE_RATIO", 0.45))
                cam_drag_threshold = max(25.0, palm_size * drag_dist_ratio)
                screen_drag_threshold = max(60.0, float(getattr(config, "MOUSE_DRAG_SCREEN_THRESHOLD", 80.0)))

                is_deliberate_drag = (held_duration >= 0.15 and dist_cam_from_down > cam_drag_threshold and dist_from_down > screen_drag_threshold)

                if (held_duration >= DRAG_HOLD_TIME or is_deliberate_drag) and not self.is_dragging:
                    try:
                        pyautogui.mouseDown()
                    except Exception:
                        pass
                    self.is_dragging = True
                    self.is_mouse_down = True
                    self.set_action("DANG KEO THA (Dragging...)", (0, 165, 255))

                if self.is_dragging:
                    self.locked_x = None
                    self.locked_y = None
                    calc_x = filtered_x
                    calc_y = filtered_y
                    cv2.line(frame, (int(tx), int(ty)), (int(target_pt[0]), int(target_pt[1])), (0, 165, 255), 3, cv2.LINE_AA)
                    cv2.circle(frame, (mid_x, mid_y), 8, (0, 165, 255), cv2.FILLED)
                    put_text_outlined(frame, "KEO THA", (mid_x - 32, mid_y - 16),
                                      scale=0.45, color=(0, 165, 255), thickness=1, outline_thickness=2)
                else:
                    # Pinch Position Lock
                    if self.locked_x is None:
                        self.locked_x = self.prev_mouse_x
                        self.locked_y = self.prev_mouse_y
                    calc_x = self.locked_x
                    calc_y = self.locked_y

                    progress = min(1.0, max(0.0, held_duration / DRAG_HOLD_TIME))
                    cv2.circle(frame, (mid_x, mid_y), 16, (60, 60, 60), 2)
                    cv2.ellipse(frame, (mid_x, mid_y), (16, 16), -90, 0, int(progress * 360), (0, 215, 255), 2)
                    put_text_outlined(frame, f"CLICK / GIU KEO {int(progress * 100)}%", (mid_x - 68, mid_y - 22),
                                      scale=0.42, color=(0, 255, 255), thickness=1, outline_thickness=2)

            else:
                if self.is_dragging:
                    if touch_ratio > DROP_RELEASE:
                        try:
                            pyautogui.mouseUp()
                        except Exception:
                            pass
                        self.is_dragging = False
                        self.is_mouse_down = False
                        self.down_start_time = None
                        self.down_pos = (0.0, 0.0)
                        self.down_cam_pos = (0.0, 0.0)
                        self.locked_x = None
                        self.locked_y = None
                        self.set_action("DA THA CHUOT (Dropped)", (0, 255, 100))
                    else:
                        cv2.line(frame, (int(tx), int(ty)), (int(target_pt[0]), int(target_pt[1])), (0, 165, 255), 3, cv2.LINE_AA)
                        cv2.circle(frame, (mid_x, mid_y), 8, (0, 165, 255), cv2.FILLED)
                        put_text_outlined(frame, "KEO THA", (mid_x - 32, mid_y - 16),
                                          scale=0.45, color=(0, 165, 255), thickness=1, outline_thickness=2)
                    calc_x = filtered_x
                    calc_y = filtered_y
                else:
                    if touch_ratio > TOUCH_RELEASE:
                        if self.down_start_time is not None:
                            if curr_time - self.last_left_click_time >= self.left_click_cooldown:
                                try:
                                    pyautogui.click()
                                except Exception:
                                    pass
                                self.last_left_click_time = curr_time
                                self.set_action("CLICK TRAI (Ap ma ngon tro)", (0, 255, 0))
                        self.down_start_time = None
                        self.down_pos = (0.0, 0.0)
                        self.down_cam_pos = (0.0, 0.0)
                        self.locked_x = None
                        self.locked_y = None

                    if touch_ratio < TOUCH_LOCK:
                        if self.locked_x is None:
                            self.locked_x = self.prev_mouse_x
                            self.locked_y = self.prev_mouse_y
                        calc_x = self.locked_x
                        calc_y = self.locked_y
                    else:
                        self.locked_x = None
                        self.locked_y = None
                        if self.deadzone > 0.0 and dist_moved < self.deadzone:
                            calc_x = self.prev_mouse_x
                            calc_y = self.prev_mouse_y
                        else:
                            calc_x = filtered_x
                            calc_y = filtered_y

            calc_x = max(2, min(self.screen_w - 2, int(calc_x)))
            calc_y = max(2, min(self.screen_h - 2, int(calc_y)))

            if calc_x != int(self.prev_mouse_x) or calc_y != int(self.prev_mouse_y):
                try:
                    pyautogui.moveTo(calc_x, calc_y)
                except Exception:
                    pass
                self.prev_mouse_x = float(calc_x)
                self.prev_mouse_y = float(calc_y)

            cursor_col = (0, 165, 255) if self.is_dragging else ((0, 255, 255) if self.locked_x is not None else (0, 255, 0))
            cv2.circle(frame, (int(ix), int(iy)), 8, cursor_col, cv2.FILLED)
            cv2.circle(frame, (int(ix), int(iy)), 14, cursor_col, 2)

        self._draw_action_hud(frame)

    # ── Vẽ HUD bổ trợ ─────────────────────────────────────────────────────────

    def _draw_control_box(self, frame: np.ndarray) -> None:
        """Vẽ vùng khung giới hạn điều khiển chuột."""
        x1 = self.margin_x
        y1 = self.margin_y
        x2 = self.frame_w - self.margin_x
        y2 = self.frame_h - self.margin_y

        line_len = 25
        col = (80, 80, 80)
        thick = 2

        cv2.line(frame, (x1, y1), (x1 + line_len, y1), col, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), col, thick)
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), col, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), col, thick)
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), col, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), col, thick)
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), col, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), col, thick)

        cv2.putText(frame, "Vung chuot", (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    def _draw_slider(
        self, frame: np.ndarray, title: str, percent: int, color: Tuple[int, int, int], x_pos: int = 30
    ) -> None:
        """Vẽ thanh trượt đứng hiển thị giá trị Volume hoặc Brightness."""
        h, w = frame.shape[:2]
        bar_y1 = 120
        bar_y2 = h - 60
        bar_w = 20

        draw_rect_transparent(frame, x_pos - 6, bar_y1 - 25, x_pos + bar_w + 6, bar_y2 + 30,
                              color=(20, 20, 20), alpha=0.6)
        cv2.rectangle(frame, (x_pos, bar_y1), (x_pos + bar_w, bar_y2), (60, 60, 60), -1)

        fill_h = int((percent / 100.0) * (bar_y2 - bar_y1))
        fill_y = bar_y2 - fill_h
        cv2.rectangle(frame, (x_pos, fill_y), (x_pos + bar_w, bar_y2), color, -1)

        put_text_outlined(frame, title, (x_pos - 4, bar_y1 - 8),
                          scale=0.45, color=color, thickness=1, outline_thickness=2)
        put_text_outlined(frame, f"{percent}%", (x_pos - 4, bar_y2 + 20),
                          scale=0.45, color=color, thickness=1, outline_thickness=2)

    def _draw_action_hud(self, frame: np.ndarray) -> None:
        """Vẽ thông báo hành động vừa kích hoạt ở góc dưới trái."""
        if time.time() - self.action_time < 1.2 and self.action_text:
            put_text_outlined(frame, f"Lenh: {self.action_text}", (20, self.frame_h - 25),
                              scale=0.7, color=self.action_color, thickness=2, outline_thickness=4)
