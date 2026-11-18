"""
src/modes/mode2_presentation.py
================================
Chế độ 2: Trình chiếu Slide & Điều khiển Thuyết trình Chuyên Nghiệp (Presentation Mode).

Cơ chế điều khiển tối ưu theo yêu cầu công thái học:
1. Phím nóng Chữ L (L-Sign Hold 1.0s):
   - Ngón cái và ngón trỏ duỗi vuông góc hình chữ L, 3 ngón còn lại gập.
   - Giữ 1.0s kèm vòng tiến trình Radial Ring -> Gửi phím F5 (Bắt đầu / Bật trình chiếu).
2. Quy trình Chuyển Slide An Toàn (Arm-then-Swipe Workflow):
   - Giơ 1 bàn tay mở lên giữ yên trong 1.0 giây để nạp trạng thái (ARMING -> ARMED).
   - Khi đã SẴN SÀNG (ARMED), quơ tay dứt khoát:
     * Quơ sang trái (Swipe Left) -> Next Slide (phím Right).
     * Quơ sang phải (Swipe Right) -> Prev Slide (phím Left).
   - Tự động khóa đệm Cooldown 0.8s và reset về IDLE sau mỗi lần chuyển slide,
     ngăn chặn 100% hiện tượng chuyển nhầm 2 slide hoặc giật lùi khi thu tay về.
3. HUD hiển thị đồ họa Neon chuyên nghiệp, trực quan trên khung hình camera.
"""

import time
import math
import collections
import queue
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import cv2

import config


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
    """Vẽ chữ có viền đen để tương phản sắc nét trên mọi khung hình camera."""
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
    color: Tuple[int, int, int] = (20, 25, 40),
    alpha: float = 0.65,
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


# ── Bộ điều khiển PresentationController ──────────────────────────────────────

class PresentationController:
    """
    Bộ điều khiển chuyên biệt cho Mode 2 (Trình chiếu Slide).
    """

    def __init__(
        self,
        frame_w: int = getattr(config, "CAMERA_WIDTH", 1280),
        frame_h: int = getattr(config, "CAMERA_HEIGHT", 720),
        action_queue: Optional[queue.Queue] = None,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.action_queue = action_queue

        # ── Cấu hình ngưỡng thời gian ──────────────────────────────────────────
        self.ARM_HOLD_DURATION = 1.0       # Giữ tay 1.0s để nạp sẵn sàng quơ
        self.L_HOLD_DURATION = 1.0         # Giữ chữ L 1.0s để kích hoạt F5
        self.SWIPE_COOLDOWN = 0.8          # Thời gian hồi sau mỗi lần quơ (giây)
        self.L_COOLDOWN = 1.5              # Thời gian hồi sau khi bật F5 (giây)

        # ── Trạng thái Quơ chuyển Slide (Arm-then-Swipe State) ─────────────────
        # State: IDLE -> ARMING -> ARMED -> COOLDOWN
        self.slide_state = "IDLE"
        self.arm_start_time: Optional[float] = None
        self.arm_pos: Optional[Tuple[float, float]] = None      # Tọa độ mốc neo khi nạp
        self.armed_time: Optional[float] = None
        self.last_swipe_time = 0.0

        # Lịch sử vị trí gần nhất để phát hiện hướng di chuyển tức thời
        self.recent_positions = collections.deque(maxlen=6)

        # ── Trạng thái Phím nóng chữ L (F5) ───────────────────────────────────
        self.l_sign_start_time: Optional[float] = None
        self.last_l_trigger_time = 0.0

        # ── Thông báo HUD ─────────────────────────────────────────────────────
        self.action_text = ""
        self.action_color = (0, 220, 255)
        self.action_display_until = 0.0

    def set_frame_dimensions(self, frame_w: int, frame_h: int) -> None:
        """Cập nhật kích thước khung hình camera khi thay đổi độ phân giải."""
        self.frame_w = frame_w
        self.frame_h = frame_h

    # ── Các hàm tính toán hình thái học bàn tay ──────────────────────────────

    @staticmethod
    def _get_coords(pt: Any) -> Tuple[float, float]:
        """Trích xuất (x, y) từ landmark tuple hoặc object."""
        if hasattr(pt, "x") and hasattr(pt, "y"):
            return float(pt.x), float(pt.y)
        if isinstance(pt, dict):
            return float(pt.get("x", 0.0)), float(pt.get("y", 0.0))
        if len(pt) >= 4:
            return float(pt[1]), float(pt[2])
        if len(pt) >= 2:
            return float(pt[0]), float(pt[1])
        return 0.0, 0.0

    def get_palm_size(self, lm_list: Any) -> float:
        """Kích thước bàn tay: khoảng cách từ Cổ tay (0) đến Khớp gốc ngón giữa (9)."""
        if not lm_list or len(lm_list) < 21:
            return 100.0
        w_x, w_y = self._get_coords(lm_list[config.WRIST])
        m_x, m_y = self._get_coords(lm_list[config.MIDDLE_MCP])
        return max(15.0, math.hypot(m_x - w_x, m_y - w_y))

    def count_fingers_up(self, lm_list: Any) -> List[int]:
        """
        Đếm trạng thái duỗi/gập của 5 ngón: [Cái, Trỏ, Giữa, Áp út, Út].
        1 = Duỗi, 0 = Gập.
        """
        if not lm_list or len(lm_list) < 21:
            return [0, 0, 0, 0, 0]

        pts = [self._get_coords(lm_list[i]) for i in range(21)]
        palm_size = self.get_palm_size(lm_list)
        fingers = [0, 0, 0, 0, 0]

        # 4 ngón: Trỏ, Giữa, Áp út, Út (So sánh TIP với PIP)
        # Vì ảnh đã flip nên toạ độ Y: hướng lên trên là Y nhỏ hơn
        tips = [config.INDEX_TIP, config.MIDDLE_TIP, config.RING_TIP, config.PINKY_TIP]
        pips = [config.INDEX_PIP, config.MIDDLE_PIP, config.RING_PIP, config.PINKY_PIP]
        mcps = [config.INDEX_MCP, config.MIDDLE_MCP, config.RING_MCP, config.PINKY_MCP]

        for idx, (t, p, m) in enumerate(zip(tips, pips, mcps)):
            # Ngón duỗi khi TIP cao hơn PIP và khoảng cách TIP-MCP đủ lớn
            if pts[t][1] < pts[p][1] and math.hypot(pts[t][0] - pts[m][0], pts[t][1] - pts[m][1]) > 0.45 * palm_size:
                fingers[idx + 1] = 1

        # Ngón cái: kiểm tra khoảng cách TIP ngón cái đến khớp gốc ngón út hoặc ngón trỏ
        thumb_tip = pts[config.THUMB_TIP]
        thumb_ip = pts[config.THUMB_IP]
        thumb_mcp = pts[config.THUMB_MCP]
        index_mcp = pts[config.INDEX_MCP]
        pinky_mcp = pts[config.PINKY_MCP]

        dist_thumb_index = math.hypot(thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1])
        if dist_thumb_index > 0.45 * palm_size:
            fingers[0] = 1

        return fingers

    def is_letter_l(self, lm_list: Any, fingers: List[int], palm_size: float) -> bool:
        """
        Nhận diện ký hiệu chữ "L":
        - Ngón cái duỗi: fingers[0] == 1
        - Ngón trỏ duỗi: fingers[1] == 1
        - 3 ngón còn lại gập chặt: fingers[2] == 0, fingers[3] == 0, fingers[4] == 0
        - Khoảng cách đầu ngón cái và ngón trỏ đủ lớn (tránh chụm ngón).
        - Góc giữa ngón cái và ngón trỏ tạo góc mở chữ L (~30° đến ~135° từ cổ tay).
        """
        if fingers != [1, 1, 0, 0, 0]:
            return False

        pts = [self._get_coords(lm_list[i]) for i in range(21)]
        tx, ty = pts[config.THUMB_TIP]
        ix, iy = pts[config.INDEX_TIP]
        wx, wy = pts[config.WRIST]

        # Tránh chụm ngón cái và trỏ (Pinch)
        dist_tips = math.hypot(tx - ix, ty - iy)
        if dist_tips < 0.40 * palm_size:
            return False

        # Vector ngón cái và vector ngón trỏ từ cổ tay
        v_thumb = (tx - wx, ty - wy)
        v_index = (ix - wx, iy - wy)

        mag_t = math.hypot(v_thumb[0], v_thumb[1])
        mag_i = math.hypot(v_index[0], v_index[1])
        if mag_t < 1e-4 or mag_i < 1e-4:
            return False

        dot = v_thumb[0] * v_index[0] + v_thumb[1] * v_index[1]
        cos_ang = max(-1.0, min(1.0, dot / (mag_t * mag_i)))
        angle_deg = math.degrees(math.acos(cos_ang))

        # Kiểm tra góc mở chữ L tự nhiên
        return 30.0 <= angle_deg <= 135.0

    def set_action_badge(self, text: str, color: Tuple[int, int, int] = (0, 220, 255), duration: float = 1.2) -> None:
        """Cập nhật huy hiệu hành động hiển thị trên khung hình camera."""
        self.action_text = text
        self.action_color = color
        self.action_display_until = time.time() + duration

    def _fire_action(self, action_name: str, badge_text: str, badge_color: Tuple[int, int, int]) -> None:
        """Gửi hành động vào hàng đợi đa luồng ActionDispatcher."""
        if self.action_queue is not None:
            try:
                self.action_queue.put_nowait({"action": action_name, "payload": {}})
            except queue.Full:
                pass
        self.set_action_badge(badge_text, badge_color)

    # ── Vòng lặp xử lý chính (process) ──────────────────────────────────────

    def process(self, frame: np.ndarray, raw_hands_data: List[Any]) -> None:
        """
        Xử lý khung hình mỗi frame trong Mode 2:
        - Nhận diện phím nóng chữ L (F5 Trình chiếu).
        - Quản lý quy trình: Giơ tay 1s -> Sẵn sàng -> Quơ trái/phải chuyển slide.
        """
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            fh, fw = frame.shape[:2]
            if fw != self.frame_w or fh != self.frame_h:
                self.set_frame_dimensions(fw, fh)

        now = time.time()

        # Nếu không có bàn tay nào trong khung hình
        if not raw_hands_data or len(raw_hands_data) == 0:
            self._reset_state()
            self._draw_hud(frame, hand_center=None, palm_size=100.0)
            return

        # Chỉ lấy bàn tay chính đầu tiên
        hand_pts = raw_hands_data[0]
        if not hand_pts or len(hand_pts) < 21:
            self._reset_state()
            self._draw_hud(frame, hand_center=None, palm_size=100.0)
            return

        palm_size = self.get_palm_size(hand_pts)
        fingers = self.count_fingers_up(hand_pts)
        pts = [self._get_coords(hand_pts[i]) for i in range(21)]

        # Tọa độ tâm lòng bàn tay (ước lượng qua trung bình Cổ tay và MCP ngón giữa)
        wx, wy = pts[config.WRIST]
        mx, my = pts[config.MIDDLE_MCP]
        palm_center = ((wx + mx) / 2.0, (wy + my) / 2.0)
        self.recent_positions.append((now, palm_center[0], palm_center[1]))

        # ── 1. KIỂM TRA PHÍM NÓNG CHỮ L (F5 BẮT ĐẦU TRÌNH CHIẾU) ────────────
        is_l = self.is_letter_l(hand_pts, fingers, palm_size)

        if is_l:
            # Hủy quy trình quơ slide nếu đang làm
            self.slide_state = "IDLE"
            self.arm_start_time = None
            self.arm_pos = None

            if now - self.last_l_trigger_time > self.L_COOLDOWN:
                if self.l_sign_start_time is None:
                    self.l_sign_start_time = now

                elapsed = now - self.l_sign_start_time
                progress = min(1.0, elapsed / self.L_HOLD_DURATION)

                # Vẽ vòng tiến trình chữ L
                tx, ty = pts[config.THUMB_TIP]
                ix, iy = pts[config.INDEX_TIP]
                mid_l = (int((tx + ix) / 2), int((ty + iy) / 2))

                cv2.circle(frame, mid_l, 26, (40, 50, 70), 2)
                cv2.ellipse(frame, mid_l, (26, 26), -90, 0, int(progress * 360), (0, 215, 255), 3, cv2.LINE_AA)
                put_text_outlined(frame, f"F5 TRINH CHIEU {int(progress * 100)}%",
                                  (mid_l[0] - 80, mid_l[1] - 34),
                                  scale=0.46, color=(0, 215, 255), thickness=1, outline_thickness=3)

                if elapsed >= self.L_HOLD_DURATION:
                    # Kích hoạt F5 Bắt đầu trình chiếu
                    self._fire_action("PRESENTATION_START", "DA BAT F5 TRINH CHIEU (Slide Show)", (0, 255, 128))
                    self.last_l_trigger_time = now
                    self.l_sign_start_time = None
            else:
                put_text_outlined(frame, "F5 (DA KICH HOAT)",
                                  (int(palm_center[0]) - 65, int(palm_center[1]) - 30),
                                  scale=0.46, color=(160, 160, 160), thickness=1, outline_thickness=3)

            self._draw_hud(frame, hand_center=palm_center, palm_size=palm_size)
            return
        else:
            self.l_sign_start_time = None

        # ── 2. QUY TRÌNH CHUYỂN SLIDE (GIƠ 1 BÀN TAY CHỜ 1S -> QUƠ TRÁI/PHẢI) ──

        # Kiểm tra đang trong thời gian hồi Cooldown sau khi quơ
        if now - self.last_swipe_time < self.SWIPE_COOLDOWN:
            self.slide_state = "COOLDOWN"
            cooldown_left = self.SWIPE_COOLDOWN - (now - self.last_swipe_time)
            put_text_outlined(frame, f"KHOA DEM ({cooldown_left:.1f}s)...",
                              (int(palm_center[0]) - 60, int(palm_center[1]) - 25),
                              scale=0.44, color=(120, 120, 120), thickness=1, outline_thickness=2)
            self._draw_hud(frame, hand_center=palm_center, palm_size=palm_size)
            return

        # Điều kiện bàn tay mở để nạp chuyển slide:
        # Ít nhất 3 ngón tay duỗi (thường là cả bàn tay 4-5 ngón xòe tự nhiên)
        is_open_hand = sum(fingers) >= 3

        if is_open_hand:
            # Giai đoạn 1: Đang nạp chờ 1 giây (ARMING)
            if self.slide_state in ("IDLE", "COOLDOWN"):
                self.slide_state = "ARMING"
                self.arm_start_time = now
                self.arm_pos = palm_center

            if self.slide_state == "ARMING":
                # Kiểm tra tay có đứng yên tương đối không
                if self.arm_pos is not None:
                    drift = math.hypot(palm_center[0] - self.arm_pos[0], palm_center[1] - self.arm_pos[1])
                    if drift > 0.40 * palm_size:
                        # Di chuyển quá nhanh khi đang nạp -> reset mốc neo
                        self.arm_pos = palm_center
                        self.arm_start_time = now

                elapsed = now - (self.arm_start_time or now)
                progress = min(1.0, elapsed / self.ARM_HOLD_DURATION)

                # Vẽ vòng tiến trình nạp 1.0s ngay tại tâm bàn tay
                cx, cy = int(palm_center[0]), int(palm_center[1])
                cv2.circle(frame, (cx, cy), 30, (40, 50, 70), 2)
                cv2.ellipse(frame, (cx, cy), (30, 30), -90, 0, int(progress * 360), (0, 220, 255), 3, cv2.LINE_AA)
                put_text_outlined(frame, f"GIU TAY 1s DE QUO... {int(progress * 100)}%",
                                  (cx - 105, cy - 38),
                                  scale=0.46, color=(0, 220, 255), thickness=1, outline_thickness=3)

                if elapsed >= self.ARM_HOLD_DURATION:
                    # Đã nạp thành công -> SẴN SÀNG QUƠ (ARMED)
                    self.slide_state = "ARMED"
                    self.armed_time = now
                    self.arm_pos = palm_center
                    self.set_action_badge("SAN SANG: QUO TRAI / PHAI ⬅ ➡", (0, 255, 128), duration=1.0)

            # Giai đoạn 2: Đã sẵn sàng quơ (ARMED)
            elif self.slide_state == "ARMED":
                cx, cy = int(palm_center[0]), int(palm_center[1])
                anchor_x, anchor_y = self.arm_pos if self.arm_pos else palm_center

                # Vẽ chỉ báo SẴN SÀNG QUƠ phát sáng Neon xanh lục
                cv2.circle(frame, (int(anchor_x), int(anchor_y)), 10, (0, 255, 128), 2)
                cv2.circle(frame, (cx, cy), 28, (0, 255, 128), 3, cv2.LINE_AA)
                cv2.line(frame, (int(anchor_x), int(anchor_y)), (cx, cy), (0, 255, 128), 2, cv2.LINE_AA)

                # Mũi tên hướng dẫn 2 bên
                put_text_outlined(frame, "⬅ QUO PHAI (PREV)  |  QUO TRAI (NEXT) ➡",
                                  (cx - 160, cy - 40),
                                  scale=0.48, color=(0, 255, 128), thickness=1, outline_thickness=3)

                # Tính toán độ dịch chuyển ngang và vận tốc vuốt
                delta_x = palm_center[0] - anchor_x
                swipe_threshold = max(40.0, 0.35 * palm_size)

                # Tính vận tốc vuốt tức thời qua 4 frame gần nhất
                vx = 0.0
                if len(self.recent_positions) >= 3:
                    p_old = self.recent_positions[0]
                    p_cur = self.recent_positions[-1]
                    dt = max(0.001, p_cur[0] - p_old[0])
                    vx = (p_cur[1] - p_old[1]) / dt

                # Quơ sang trái (delta_x < -ngưỡng hoặc vận tốc âm nhanh)
                if delta_x < -swipe_threshold or vx < -350.0:
                    self._fire_action("SWIPE_LEFT", "➡ SLIDE TIEP THEO (NEXT)", (0, 255, 0))
                    self.last_swipe_time = now
                    self.slide_state = "COOLDOWN"
                    self.arm_pos = None
                    self.arm_start_time = None

                # Quơ sang phải (delta_x > ngưỡng hoặc vận tốc dương nhanh)
                elif delta_x > swipe_threshold or vx > 350.0:
                    self._fire_action("SWIPE_RIGHT", "⬅ SLIDE TRUOC (PREV)", (0, 200, 255))
                    self.last_swipe_time = now
                    self.slide_state = "COOLDOWN"
                    self.arm_pos = None
                    self.arm_start_time = None

                # Hết thời gian chờ quơ (nếu giơ tay quá 4.5s mà không quơ -> tự thoát armed)
                elif now - (self.armed_time or now) > 4.5:
                    self.slide_state = "IDLE"
                    self.arm_pos = None
                    self.arm_start_time = None

        else:
            # Người dùng hạ tay hoặc gập tay -> reset về IDLE
            if self.slide_state in ("ARMING", "ARMED"):
                self.slide_state = "IDLE"
                self.arm_start_time = None
                self.arm_pos = None

        self._draw_hud(frame, hand_center=palm_center, palm_size=palm_size)

    def _reset_state(self) -> None:
        """Reset toàn bộ trạng thái khi mất dấu bàn tay."""
        self.slide_state = "IDLE"
        self.arm_start_time = None
        self.arm_pos = None
        self.l_sign_start_time = None
        self.recent_positions.clear()

    # ── Vẽ HUD đồ họa Mode 2 ────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, hand_center: Optional[Tuple[float, float]], palm_size: float) -> None:
        """Vẽ thanh thông tin trạng thái Trình chiếu Mode 2 ở đầu khung hình."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return

        h, w = frame.shape[:2]
        now = time.time()

        # Thanh banner hiển thị huy hiệu hành động (Action Badge)
        if self.action_text and now < self.action_display_until:
            draw_rect_transparent(frame, w // 2 - 210, 20, w // 2 + 210, 68, color=(15, 23, 42), alpha=0.75)
            cv2.rectangle(frame, (w // 2 - 210, 20), (w // 2 + 210, 68), self.action_color, 2)
            put_text_outlined(frame, self.action_text,
                              (w // 2 - 185, 52),
                              scale=0.62, color=self.action_color, thickness=2, outline_thickness=3)
        else:
            # Thanh hướng dẫn mặc định của Mode 2
            draw_rect_transparent(frame, w // 2 - 240, 20, w // 2 + 240, 60, color=(15, 23, 42), alpha=0.65)
            cv2.rectangle(frame, (w // 2 - 240, 20), (w // 2 + 240, 60), (99, 102, 241), 1)
            put_text_outlined(frame, "MODE 2: TRINH CHIEU | GIO TAY 1s -> QUO DOI SLIDE | CHU L: F5",
                              (w // 2 - 225, 46),
                              scale=0.45, color=(220, 220, 255), thickness=1, outline_thickness=2)
