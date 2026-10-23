"""
src/inference/voting_smoother.py
================================
Bộ lọc làm mượt suy luận thời gian thực (5-Frame Majority Voting + Cooldown State Machine).

Bản chất kỹ thuật:
1. Vấn đề "Nhấp nháy nhãn" (Label Flickering) và "Kích hoạt lặp" (Duplicate Firings):
   - Mạng Bi-LSTM chạy suy luận trên cửa sổ trượt 30 frame mỗi khi có frame mới (30 lần/giây).
   - Một cử chỉ vuốt kéo dài ~15 frame sẽ khiến mạng dự đoán "SWIPE_LEFT" liên tiếp 15 lần.
   - Nếu không xử lý, hệ thống sẽ chuyển liên tiếp 15 slide PowerPoint!

2. Giải pháp 2 tầng:
   a) Tầng 1: Lọc biểu quyết đa số (Majority Voting):
      - Lưu lịch sử N frame gần nhất (mặc định N = 5).
      - Chỉ công nhận nhãn khi có >= 60% số frame trong cửa sổ đồng thuận và xác suất Softmax >= 80%.
   b) Tầng 2: Máy trạng thái có thời gian khóa đệm (Cooldown State Machine):
      - Phân loại cử chỉ thành 2 nhóm:
        + Cử chỉ rời rạc (Discrete: SWIPE, CLICK, PINCH): Kích hoạt 1 lần duy nhất, sau đó khóa (Lockout Cooldown = 0.6s)
          cho đến khi người dùng hạ tay hoặc đưa tay về trạng thái IDLE.
        + Cử chỉ liên tục (Continuous: SCROLL, VOLUME): Cho phép dòng dữ liệu (stream) được cập nhật liên tục mượt mà.
"""

import time
import collections
from typing import Tuple, Optional, List, Dict


class VotingSmoother:
    """Bộ lọc đa số và điều phối trạng thái cử chỉ."""

    # Phân nhóm cử chỉ rời rạc (cần khóa cooldown chống kích hoạt lặp)
    DISCRETE_GESTURES = {
        "PINCH_CLICK",
        "SWIPE_LEFT",
        "SWIPE_RIGHT",
        "SWIPE_UP",
        "SWIPE_DOWN",
        "FIST_LOCK",
        "OPEN_PALM",
    }

    # Phân nhóm cử chỉ liên tục (cho phép stream mượt mà)
    CONTINUOUS_GESTURES = {
        "TWO_HAND_ZOOM_IN",
        "TWO_HAND_ZOOM_OUT",
    }

    def __init__(
        self,
        window_size: int = 5,
        min_ratio: float = 0.60,
        confidence_threshold: float = 0.80,
        cooldown_sec: float = 0.60,
        continuous_interval_sec: float = 0.15,
    ):
        self.window_size = window_size
        self.min_ratio = min_ratio
        self.confidence_threshold = confidence_threshold
        self.cooldown_sec = cooldown_sec
        self.continuous_interval_sec = continuous_interval_sec

        self.history = collections.deque(maxlen=window_size)
        self.last_triggered_gesture = "IDLE"
        self.last_trigger_times: Dict[str, float] = {}
        self.is_gesture_active: bool = False  # Edge-trigger flag: Yêu cầu thả về IDLE trước khi kích hoạt lại

    def update(self, raw_class_name: str, confidence: float) -> Tuple[Optional[str], float, str]:
        """
        Cập nhật dự đoán frame hiện tại.
        Trả về: (action_to_fire, confidence, current_stable_label)
        - action_to_fire: Tên cử chỉ cần kích hoạt (None nếu đang trong cooldown hoặc chưa đủ tin cậy).
        - confidence: Độ tin cậy của nhãn ổn định.
        - current_stable_label: Nhãn đại diện của cửa sổ 5 frame hiện tại.
        """
        curr_time = time.time()

        # 1. Nếu độ tin cậy quá thấp, gán là IDLE để không gây nhiễu
        effective_class = raw_class_name if confidence >= self.confidence_threshold else "IDLE"
        self.history.append((effective_class, confidence))

        if len(self.history) < self.window_size:
            return None, confidence, "WAITING"

        # 2. Đếm tần suất xuất hiện của các nhãn trong cửa sổ
        counts: Dict[str, int] = collections.defaultdict(int)
        conf_sum: Dict[str, float] = collections.defaultdict(float)

        for c_name, conf in self.history:
            counts[c_name] += 1
            conf_sum[c_name] += conf

        # Tìm nhãn chiếm đa số
        most_common_class, most_common_count = max(counts.items(), key=lambda item: item[1])
        ratio = most_common_count / float(self.window_size)
        avg_conf = conf_sum[most_common_class] / max(1, most_common_count)

        # Nếu không đạt tỷ lệ đa số (>60%) hoặc là IDLE
        if ratio < self.min_ratio or most_common_class == "IDLE":
            self.is_gesture_active = False  # Đã thả tay về IDLE -> Tái kích hoạt sườn (Re-arm Edge Trigger)
            return None, avg_conf, "IDLE"

        stable_label = most_common_class

        # 3. Xử lý logic Cooldown State Machine (Edge-Triggered & Per-Gesture Cooldown)
        if stable_label in self.DISCRETE_GESTURES:
            last_time = self.last_trigger_times.get(stable_label, 0.0)
            time_since_last = curr_time - last_time

            # Chỉ kích hoạt khi: Chưa bị khóa giữ nguyên tay (is_gesture_active=False) VÀ đã qua cooldown của cử chỉ này
            if not self.is_gesture_active and time_since_last >= self.cooldown_sec:
                self.last_trigger_times[stable_label] = curr_time
                self.last_triggered_gesture = stable_label
                self.is_gesture_active = True  # Đánh dấu đang giữ tay -> Khóa auto-repeat
                return stable_label, avg_conf, stable_label
            elif self.is_gesture_active:
                # Đang giữ nguyên tay sau khi kích hoạt -> Không bắn lặp lại (Chống Auto-Repeat)
                return None, avg_conf, f"{stable_label} (HOLDING)"
            else:
                # Đang trong thời gian khóa đệm chống lặp
                return None, avg_conf, f"{stable_label} (LOCKOUT)"

        elif stable_label in self.CONTINUOUS_GESTURES:
            # Nếu là cử chỉ liên tục: Cho phép bắn với tần số kiểm soát (Rate Limiting)
            last_time = self.last_trigger_times.get(stable_label, 0.0)
            if curr_time - last_time >= self.continuous_interval_sec:
                self.last_trigger_times[stable_label] = curr_time
                self.last_triggered_gesture = stable_label
                return stable_label, avg_conf, stable_label
            else:
                return None, avg_conf, stable_label

        return None, avg_conf, stable_label

    def reset(self):
        """Khởi động lại bộ lọc."""
        self.history.clear()
        self.last_triggered_gesture = "IDLE"
        self.last_trigger_times.clear()
        self.is_gesture_active = False
