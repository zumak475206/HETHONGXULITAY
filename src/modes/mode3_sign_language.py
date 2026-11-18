"""
src/modes/mode3_sign_language.py
=================================
Chế độ 3: Nhận Diện Ngôn Ngữ Ký Hiệu Chuyên Nghiệp (VSL Sign Language Mode).

Tính năng cốt lõi:
1. Trực quan hóa toàn diện Xử lý ảnh & Trích xuất đặc trưng (Computer Vision & AI Visualizer):
   - Khung xương 2 bàn tay MediaPipe 3D phân biệt màu sắc tương phản cao (Cyan / Green).
   - Đo đạc góc gập các khớp ngón tay và khoảng cách liên tay thời gian thực.
   - Attention Heatmap (Trọng số chú ý của mạng Bi-LSTM trên chuỗi 45 frames).
   - Bảng Top-3 xác suất nhận diện (Top-3 Probability Bars).
2. Hệ thống Nhận diện Ký hiệu & Phụ đề Hội thoại (Sentence Builder):
   - Chuyển đổi tên mã ký hiệu (VD: VSL_HELLO) sang Ý NGHĨA TIẾNG VIỆT có dấu (VD: "Xin chào").
   - Bộ ghép câu (Sentence Accumulator): Ghép các từ liên tiếp thành câu hoàn chỉnh.
   - Chống kích hoạt lặp (Word Debouncing) và tự động xóa sau 4 giây ngừng thao tác.
   - Cung cấp trạng thái phụ đề tức thời cho Cửa sổ phụ đề toàn màn hình máy tính (DesktopSubtitleOverlay).
"""

import time
import math
import collections
import queue
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import cv2

import config


# ── Tiện ích đồ họa HUD ────────────────────────────────────────────────────────

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
    color: Tuple[int, int, int] = (15, 20, 35),
    alpha: float = 0.70,
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


# ── Bộ điều khiển Chế độ 3 (SignLanguageController) ──────────────────────────

class SignLanguageController:
    """
    Bộ điều khiển chuyên biệt cho Mode 3: Nhận diện Ngôn ngữ ký hiệu VSL & Phụ đề máy tính.
    """

    def __init__(
        self,
        frame_w: int = getattr(config, "CAMERA_WIDTH", 1280),
        frame_h: int = getattr(config, "CAMERA_HEIGHT", 720),
        db: Optional[Any] = None,
        action_queue: Optional[queue.Queue] = None,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.db = db
        if self.db is None:
            try:
                from src.database.db_manager import DatabaseManager
                self.db = DatabaseManager()
            except Exception:
                self.db = None
        self.action_queue = action_queue

        # ── Cấu hình ngưỡng nhận diện ──────────────────────────────────────────
        self.CONFIDENCE_THRESHOLD = 0.50   # Ngưỡng tin cậy tối thiểu để chấp nhận từ (nhạy và bắt nhanh)
        self.WORD_HOLD_DURATION = 0.45     # Thời gian giữ cử chỉ ổn định (giây)
        self.WORD_COOLDOWN_SEC = 0.90      # Thời gian hồi giữa 2 lần nhận cùng 1 từ (giây)
        self.SENTENCE_TIMEOUT_SEC = 5.00   # Tự động xóa câu sau 5s không có thao tác

        # ── Bản đồ từ điển ý nghĩa tiếng Việt ──────────────────────────────────
        # Map: gesture_name -> display_name (Ý nghĩa từ)
        self.gesture_meanings: Dict[str, str] = {}
        self.reload_gesture_meanings()

        # ── Trạng thái nhận diện & Bộ ghép câu (Sentence Builder) ───────────────
        self.current_raw_label: str = "IDLE"
        self.current_confidence: float = 0.0
        self.current_meaning: str = ""
        self.top3_predictions: List[Tuple[str, str, float]] = []  # [(name, meaning, prob), ...]

        # Lịch sử nhãn để lọc đa số (Voting)
        self.label_history = collections.deque(maxlen=6)

        # Bộ ghép câu hội thoại
        self.sentence_words: List[str] = []
        self.last_word_added: str = ""
        self.last_word_time: float = 0.0
        self.last_activity_time: float = time.time()
        self._has_returned_to_idle: bool = True

        # Trọng số Attention 45 frames
        self.attention_weights = np.zeros(config.SEQUENCE_LENGTH, dtype=np.float32)

    def set_frame_dimensions(self, frame_w: int, frame_h: int) -> None:
        """Cập nhật kích thước khung hình camera khi thay đổi độ phân giải."""
        self.frame_w = frame_w
        self.frame_h = frame_h

    def reload_gesture_meanings(self) -> None:
        """Nạp danh mục tên mã và ý nghĩa tiếng Việt từ SQLite database."""
        meanings = {
            "IDLE": "Nghỉ / Tự do",
            "PINCH_CLICK": "Chụm ngón",
            "FIST_LOCK": "Nắm đấm",
            "OPEN_PALM": "Xòe 5 ngón",
            "SWIPE_LEFT": "Vuốt trái",
            "SWIPE_RIGHT": "Vuốt phải",
            "SWIPE_UP": "Hất lên",
            "SWIPE_DOWN": "Hạ xuống",
            "TWO_HAND_ZOOM_IN": "Phóng to",
            "TWO_HAND_ZOOM_OUT": "Thu nhỏ",
        }
        if self.db is None:
            try:
                from src.database.db_manager import DatabaseManager
                self.db = DatabaseManager()
            except Exception:
                pass

        if self.db is not None:
            try:
                gestures = self.db.get_all_gestures()
                for g in gestures:
                    g_name = str(g.get("name", "")).strip().upper()
                    g_disp = str(g.get("display_name", "")).strip()
                    if g_name and g_disp:
                        meanings[g_name] = g_disp
            except Exception:
                pass
        self.gesture_meanings = meanings

    def get_meaning(self, gesture_name: str) -> str:
        """Lấy ý nghĩa tiếng Việt của tên mã cử chỉ."""
        clean_name = gesture_name.strip().upper()
        return self.gesture_meanings.get(clean_name, clean_name)

    def clear_sentence(self) -> None:
        """Xóa câu phụ đề hội thoại hiện tại."""
        self.sentence_words.clear()
        self.last_word_added = ""
        self.current_meaning = ""
        self.last_activity_time = time.time()

    def get_full_sentence(self) -> str:
        """Trả về toàn bộ câu phụ đề đã ghép."""
        if not self.sentence_words:
            return ""
        return " ".join(self.sentence_words)

    def get_subtitle_state(self) -> Dict[str, Any]:
        """
        Trả về trạng thái phụ đề hiện tại cho DesktopSubtitleOverlay hiển thị toàn màn hình.
        """
        now = time.time()
        # Kiểm tra tự động xóa câu nếu hết thời gian chờ
        if self.sentence_words and (now - self.last_activity_time > self.SENTENCE_TIMEOUT_SEC):
            self.sentence_words.clear()
            self.last_word_added = ""

        is_active = (now - self.last_activity_time <= self.SENTENCE_TIMEOUT_SEC) and (len(self.sentence_words) > 0 or len(self.current_meaning) > 0)
        return {
            "current_word": self.current_raw_label if self.current_raw_label != "IDLE" else "",
            "current_meaning": self.current_meaning,
            "sentence": self.get_full_sentence(),
            "confidence": self.current_confidence,
            "top3": self.top3_predictions,
            "is_active": is_active,
        }

    @staticmethod
    def get_finger_extension_state(raw_hand_pts: list) -> Dict[str, Any]:
        """
        Đo độ duỗi thực tế của từng ngón tay trong không gian 3D (bất biến xoay 360°)
        và kiểm tra tạo hình vòng tròn chữ O / số 0.
        """
        if not raw_hand_pts or len(raw_hand_pts) < 21:
            return {}
        lm = {p[0]: np.array([p[1], p[2], p[3]], dtype=np.float32) for p in raw_hand_pts}
        w = lm[0]
        res = {}
        for name, tip_i, pip_i in [('thumb', 4, 2), ('index', 8, 6), ('middle', 12, 10), ('ring', 16, 14), ('pinky', 20, 18)]:
            d_tip = np.linalg.norm(lm[tip_i] - w)
            d_pip = np.linalg.norm(lm[pip_i] - w)
            thresh = 1.15 if name != 'thumb' else 1.10
            res[name] = bool(d_tip > thresh * d_pip)

        # Đo khoảng cách đầu ngón cái (4) và đầu ngón trỏ (8) theo tỷ lệ lòng bàn tay
        palm_size = float(np.linalg.norm(lm[9] - w))
        if palm_size < 1e-4:
            palm_size = 1.0
        d_thumb_index_norm = float(np.linalg.norm(lm[4] - lm[8])) / palm_size
        res['d_thumb_index_norm'] = d_thumb_index_norm

        # Nhận diện tạo hình chữ O / Số 0 (vòng tròn khép giữa ngón cái và ngón trỏ, các ngón khác co)
        res['is_o_shape'] = bool(
            d_thumb_index_norm < 0.50 and
            not res.get('middle', False) and
            not res.get('ring', False) and
            not res.get('pinky', False)
        )
        return res

    # ── Xử lý suy luận & Ghép câu (Process) ──────────────────────────────────

    def process_inference(
        self,
        logits: np.ndarray,
        attn_weights: Optional[np.ndarray],
        class_list: List[str],
        has_hands: bool = True,
        raw_hands_data: Optional[List[Any]] = None,
    ) -> Tuple[str, float, str]:
        """
        Tiếp nhận kết quả đầu ra từ mô hình Bi-LSTM:
        - Tính xác suất Softmax.
        - Trích xuất Top-3.
        - Lọc ổn định và ghép vào câu phụ đề.
        """
        now = time.time()

        if not has_hands or logits is None or len(logits) == 0:
            self.current_raw_label = "IDLE"
            self.current_confidence = 0.0
            self.top3_predictions = []
            return "IDLE", 0.0, ""

        if attn_weights is not None and len(attn_weights) > 0:
            self.attention_weights = np.array(attn_weights, dtype=np.float32)

        # Tính Softmax ban đầu
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        # Xác thực hình học giải phẫu ngón tay (Kinematic Anatomical Grounding)
        # Ngăn chặn AI nhầm lẫn giữa số 0, số 1, số 2 và xòe 5 ngón
        if raw_hands_data and len(raw_hands_data) > 0 and len(probs) > 0:
            f_state = self.get_finger_extension_state(raw_hands_data[0])
            idx_up = f_state.get("index", False)
            mid_up = f_state.get("middle", False)
            ring_up = f_state.get("ring", False)
            pinky_up = f_state.get("pinky", False)
            is_o_shape = f_state.get("is_o_shape", False)

            # Trường hợp 0: Số 0 / Chữ O (Đầu ngón cái và ngón trỏ khép vòng tròn, các ngón khác co)
            if is_o_shape:
                for idx, name in enumerate(class_list):
                    if name not in ("VSL_SO0", "SO_0", "0") and self.get_meaning(name).strip() != "0":
                        probs[idx] *= 0.01
                # Tìm tên lớp của số 0 trong danh mục lớp
                zero_classes = [c for c in class_list if c in ("VSL_SO0", "SO_0", "0") or self.get_meaning(c).strip() == "0"]
                for zc in zero_classes:
                    probs[class_list.index(zc)] = max(probs[class_list.index(zc)], 0.92)

            # Trường hợp 1: Xòe bàn tay (Open Palm: cả 4 ngón đều duỗi)
            elif idx_up and mid_up and ring_up and pinky_up:
                for name in ["VSL_SO0", "VSL_SO1", "VSL_SO2", "PINCH_CLICK"]:
                    if name in class_list:
                        probs[class_list.index(name)] *= 0.02
                if "OPEN_PALM" in class_list:
                    probs[class_list.index("OPEN_PALM")] = max(probs[class_list.index("OPEN_PALM")], 0.85)

            # Trường hợp 2: Số 2 (Chỉ ngón trỏ và ngón giữa duỗi, ngón nhẫn và út co)
            elif idx_up and mid_up and not ring_up and not pinky_up:
                for name in ["VSL_SO0", "VSL_SO1", "OPEN_PALM"]:
                    if name in class_list:
                        probs[class_list.index(name)] *= 0.05
                if "VSL_SO2" in class_list:
                    probs[class_list.index("VSL_SO2")] = max(probs[class_list.index("VSL_SO2")], 0.85)

            # Trường hợp 3: Số 1 (Chỉ ngón trỏ duỗi, ngón giữa/nhẫn/út co và KHÔNG PHẢI hình chữ O)
            elif idx_up and not mid_up and not ring_up and not pinky_up:
                for name in ["VSL_SO0", "VSL_SO2", "OPEN_PALM"]:
                    if name in class_list:
                        probs[class_list.index(name)] *= 0.05
                if "VSL_SO1" in class_list:
                    probs[class_list.index("VSL_SO1")] = max(probs[class_list.index("VSL_SO1")], 0.85)

            # Chuẩn hóa lại tổng xác suất = 1.0
            s = np.sum(probs)
            if s > 1e-6:
                probs = probs / s

        # Top 3 lớp có xác suất cao nhất
        top_indices = np.argsort(probs)[::-1][:min(3, len(probs))]
        top3 = []
        for idx in top_indices:
            cls_name = class_list[idx] if idx < len(class_list) else f"CLASS_{idx}"
            meaning = self.get_meaning(cls_name)
            top3.append((cls_name, meaning, float(probs[idx])))
        self.top3_predictions = top3

        best_idx = int(top_indices[0])
        best_name = class_list[best_idx] if best_idx < len(class_list) else "IDLE"
        best_conf = float(probs[best_idx])

        self.label_history.append((best_name, best_conf))

        # Kiểm tra điều kiện xác nhận từ:
        # Nhãn xuất hiện >= 4 lần trong 6 frame gần nhất và conf >= threshold
        recent_labels = [lbl for lbl, c in self.label_history]
        most_common, count = collections.Counter(recent_labels).most_common(1)[0]
        avg_conf = np.mean([c for lbl, c in self.label_history if lbl == most_common])

        self.current_raw_label = most_common
        self.current_confidence = float(avg_conf)

        if most_common != "IDLE" and count >= 3 and avg_conf >= self.CONFIDENCE_THRESHOLD:
            meaning = self.get_meaning(most_common)
            self.current_meaning = meaning
            self.last_activity_time = now

            # Thêm từ vào câu:
            # Chỉ thêm khi:
            # 1. Câu đang rỗng
            # 2. Hoặc từ mới khác từ vừa thêm gần nhất (meaning != self.last_word_added)
            # 3. Hoặc người dùng đã thả lỏng/hạ tay về IDLE rồi mới làm lại cùng từ này
            can_add = False
            if not self.sentence_words:
                can_add = True
            elif self.last_word_added != meaning:
                can_add = True
            elif self._has_returned_to_idle and (now - self.last_word_time > self.WORD_COOLDOWN_SEC):
                can_add = True

            if can_add:
                self.sentence_words.append(meaning)
                self.last_word_added = meaning
                self.last_word_time = now
                self._has_returned_to_idle = False

                # Giới hạn tối đa 12 từ trong 1 câu hiển thị
                if len(self.sentence_words) > 12:
                    self.sentence_words.pop(0)

        elif most_common == "IDLE":
            self._has_returned_to_idle = True

        # Kiểm tra tự động xóa câu sau timeout
        if self.sentence_words and (now - self.last_activity_time > self.SENTENCE_TIMEOUT_SEC):
            self.sentence_words.clear()
            self.current_meaning = ""

        return self.current_raw_label, self.current_confidence, self.get_full_sentence()

    # ── Vẽ HUD trực quan hóa toàn diện ───────────────────────────────────────

    def draw_hud(
        self,
        frame: np.ndarray,
        hands_data: Optional[List[Any]] = None,
        raw_hands_data: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Vẽ toàn bộ thông tin xử lý ảnh, trích xuất đặc trưng và phụ đề lên khung hình camera:
        1. Khung xương 2 bàn tay phân biệt màu (Tay 1: Cyan, Tay 2: Green).
        2. Góc gập khớp ngón & Khoảng cách liên tay.
        3. Attention Heatmap (Trọng số 45 frames).
        4. Top-3 Xác suất dự đoán.
        5. Banner Phụ Đề Tiếng Việt to rõ ở đáy khung hình.
        """
        if frame is None:
            return frame

        if hands_data is None:
            hands_data = []
        if raw_hands_data is None:
            raw_hands_data = []

        h, w = frame.shape[:2]

        # ── 1. VẼ THÔNG SỐ GÓC GẬP VÀ KHOẢNG CÁCH LIÊN TAY ─────────────────────
        num_hands = len(raw_hands_data)
        if num_hands >= 1:
            h1 = raw_hands_data[0]
            if len(h1) >= 21:
                # Tính khoảng cách cổ tay đến ngón giữa
                w_pt = (h1[config.WRIST][1], h1[config.WRIST][2])
                m_pt = (h1[config.MIDDLE_MCP][1], h1[config.MIDDLE_MCP][2])
                palm_sz = math.hypot(m_pt[0] - w_pt[0], m_pt[1] - w_pt[1])
                put_text_outlined(frame, f"TAY 1: Palm {int(palm_sz)}px",
                                  (int(w_pt[0]) - 50, int(w_pt[1]) + 25),
                                  scale=0.45, color=(0, 240, 255), thickness=1, outline_thickness=3)

        if num_hands >= 2:
            h1, h2 = raw_hands_data[0], raw_hands_data[1]
            if len(h1) >= 21 and len(h2) >= 21:
                w1 = (int(h1[config.WRIST][1]), int(h1[config.WRIST][2]))
                w2 = (int(h2[config.WRIST][1]), int(h2[config.WRIST][2]))
                inter_dist = math.hypot(w1[0] - w2[0], w1[1] - w2[1])

                # Vẽ đường nối giữa 2 cổ tay
                cv2.line(frame, w1, w2, (255, 120, 0), 2, cv2.LINE_AA)
                mid_x = (w1[0] + w2[0]) // 2
                mid_y = (w1[1] + w2[1]) // 2
                put_text_outlined(frame, f"Lien tay: {int(inter_dist)}px",
                                  (mid_x - 50, mid_y - 10),
                                  scale=0.45, color=(255, 160, 0), thickness=1, outline_thickness=3)

        # ── 2. VẼ TOP-3 XÁC SUẤT DỰ ĐOÁN (TOP-RIGHT CORNER) ──────────────────
        if self.top3_predictions:
            card_w, card_h = 240, 95
            card_x, card_y = w - card_w - 20, 20
            draw_rect_transparent(frame, card_x, card_y, card_x + card_w, card_y + card_h,
                                  color=(10, 15, 25), alpha=0.75)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 80, 110), 1)
            put_text_outlined(frame, "TOP 3 DU DOAN AI:", (card_x + 10, card_y + 18),
                              scale=0.42, color=(0, 220, 255), thickness=1, outline_thickness=2)

            for i, (name, meaning, prob) in enumerate(self.top3_predictions):
                row_y = card_y + 40 + i * 22
                bar_max_w = 75
                bar_w = int(prob * bar_max_w)
                color_bar = (0, 255, 128) if i == 0 else (120, 160, 220)

                # Tên ý nghĩa
                disp_txt = f"{meaning[:9]}:"
                put_text_outlined(frame, disp_txt, (card_x + 10, row_y),
                                  scale=0.40, color=(220, 220, 220), thickness=1, outline_thickness=2)

                # Thanh tiến trình
                cv2.rectangle(frame, (card_x + 95, row_y - 10), (card_x + 95 + bar_max_w, row_y - 2), (40, 45, 60), -1)
                cv2.rectangle(frame, (card_x + 95, row_y - 10), (card_x + 95 + bar_w, row_y - 2), color_bar, -1)

                # Phần trăm
                put_text_outlined(frame, f"{int(prob * 100)}%", (card_x + 180, row_y),
                                  scale=0.38, color=color_bar, thickness=1, outline_thickness=2)

        # ── 3. VẼ ATTENTION HEATMAP (BOTTOM-LEFT) ─────────────────────────────
        if self.attention_weights is not None and len(self.attention_weights) > 0:
            hm_x, hm_y, hm_w, hm_h = 25, h - 90, 240, 16
            draw_rect_transparent(frame, hm_x - 4, hm_y - 18, hm_x + hm_w + 4, hm_y + hm_h + 4,
                                  color=(10, 15, 25), alpha=0.75)
            put_text_outlined(frame, f"ATTENTION WEIGHTS ({len(self.attention_weights)} FRAMES)",
                              (hm_x, hm_y - 5), scale=0.38, color=(180, 200, 220), thickness=1, outline_thickness=2)

            w_arr = self.attention_weights
            w_min, w_max = float(np.min(w_arr)), float(np.max(w_arr))
            norm_w = (w_arr - w_min) / max(1e-5, (w_max - w_min))
            step = hm_w / max(1, len(norm_w))

            for i, val in enumerate(norm_w):
                bx = int(hm_x + i * step)
                bw = int(step) + 1
                color = cv2.applyColorMap(np.array([[[int(val * 255)]]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                col_bgr = (int(color[0]), int(color[1]), int(color[2]))
                cv2.rectangle(frame, (bx, hm_y), (bx + bw, hm_y + hm_h), col_bgr, cv2.FILLED)

        # ── 4. VẼ BANNER PHỤ ĐỀ NEON Ở ĐÁY KHUNG HÌNH (SUBTITLE HUD) ──────────
        sentence = self.get_full_sentence()
        if sentence or self.current_meaning:
            banner_h = 60
            banner_y1 = h - banner_h - 15
            banner_y2 = h - 15

            # Tính chiều rộng hộp phụ đề vừa vặn với nội dung
            text_to_show = sentence if sentence else self.current_meaning
            full_caption = f"[PHU DE VSL]: {text_to_show}"
            banner_w = min(w - 60, max(420, len(full_caption) * 16))
            banner_x1 = (w - banner_w) // 2
            banner_x2 = banner_x1 + banner_w

            draw_rect_transparent(frame, banner_x1, banner_y1, banner_x2, banner_y2,
                                  color=(8, 12, 24), alpha=0.85)
            cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), (0, 240, 255), 2)

            # Icon và câu phụ đề
            put_text_outlined(frame, full_caption,
                              (banner_x1 + 20, banner_y1 + 38),
                              scale=0.68, color=(0, 255, 180), thickness=2, outline_thickness=4)

        return frame
