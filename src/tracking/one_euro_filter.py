"""
src/tracking/one_euro_filter.py
================================
Bộ lọc thích ứng tốc độ One Euro Filter (1€ Filter).
Tài liệu tham khảo:
    Casiez, G., Roussel, N. and Vogel, D. (2012).
    1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
    ACM Conference on Human Factors in Computing Systems (CHI '12).

Nguyên lý hoạt động:
    - Khi người dùng giữ yên tay hoặc di chuyển chậm: Tần số cắt f_cutoff tiến gần min_cutoff,
      lọc triệt để nhiễu tần số cao (micro-jitter) do cảm biến camera/MediaPipe sinh ra.
    - Khi người dùng vung tay nhanh: Tần số cắt f_cutoff tăng tuyến tính theo tốc độ
      (f_cutoff = min_cutoff + beta * |dx_hat|), giảm độ trễ (latency) về xấp xỉ 0.
"""

import math
import time
from typing import List, Tuple, Optional, Any


class OneEuroFilter:
    """
    Bộ lọc 1D One Euro Filter cho một đại lượng vô hướng.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.008, d_cutoff: float = 1.0):
        """
        Tham số
        -------
        min_cutoff : float
            Tần số cắt tối thiểu (Hz). Giá trị càng nhỏ càng làm mượt khi đứng yên,
            nhưng nếu quá nhỏ sẽ gây cảm giác ì khi mới bắt đầu cử động. (Mặc định: 1.0)
        beta : float
            Hệ số tăng tốc độ nhạy theo vận tốc. Càng lớn thì độ trễ khi vung tay nhanh
            càng giảm về 0. (Mặc định: 0.008)
        d_cutoff : float
            Tần số cắt lọc đạo hàm vận tốc dx/dt (Hz). Làm mượt biến thiên tốc độ,
            chống giật khi đổi hướng đột ngột. (Mặc định: 1.0)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)

        self.x_prev     = None
        self.dx_prev    = 0.0
        self.t_prev     = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """
        Tính hệ số làm mượt alpha của bộ lọc thông thấp bậc 1 (Exponential Smoothing):
            alpha = (2 * pi * cutoff * dt) / (2 * pi * cutoff * dt + 1)
        """
        if cutoff <= 0.0 or dt <= 0.0:
            return 0.0
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def reset(self) -> None:
        """Reset toàn bộ trạng thái bộ lọc về ban đầu."""
        self.x_prev  = None
        self.dx_prev = 0.0
        self.t_prev  = None

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Lọc giá trị x tại thời điểm timestamp.

        Tham số
        -------
        x : float
            Giá trị đầu vào thô.
        timestamp : float, optional
            Mốc thời gian (giây), mặc định dùng time.time().

        Trả về
        ------
        float
            Giá trị đã được làm mượt.
        """
        if math.isnan(x) or math.isinf(x):
            return self.x_prev if self.x_prev is not None else 0.0

        if timestamp is None:
            timestamp = time.time()

        # Frame đầu tiên sau khi khởi tạo hoặc reset
        if self.t_prev is None:
            self.x_prev  = float(x)
            self.dx_prev = 0.0
            self.t_prev  = float(timestamp)
            return float(x)

        dt = float(timestamp - self.t_prev)
        if dt < 0.0:
            # Timestamp đảo ngược -> đồng bộ lại mốc thời gian
            self.x_prev  = float(x)
            self.dx_prev = 0.0
            self.t_prev  = float(timestamp)
            return float(x)

        if dt <= 1e-6:
            # Hai frame liên tiếp quá sát nhau hoặc timestamp trùng lặp
            return self.x_prev

        # 1. Tính và làm mượt đạo hàm (vận tốc biến thiên tín hiệu)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 2. Tần số cắt thích nghi theo độ lớn vận tốc
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. Làm mượt tín hiệu chính theo tần số cắt thích ứng
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        # 4. Cập nhật trạng thái
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = timestamp

        return x_hat


class PointFilter:
    """
    Bộ lọc 2D (x, y) sử dụng 2 OneEuroFilter độc lập cho tọa độ con trỏ chuột / màn hình.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.008, d_cutoff: float = 1.0):
        self.x_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.y_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        Lọc cặp tọa độ (x, y) tại thời điểm timestamp.

        Trả về
        ------
        (float, float) : Cặp tọa độ (filtered_x, filtered_y)
        """
        if timestamp is None:
            timestamp = time.time()

        fx = self.x_filter.filter(x, timestamp)
        fy = self.y_filter.filter(y, timestamp)
        return fx, fy

    def reset(self) -> None:
        """Reset cả 2 trục X và Y."""
        self.x_filter.reset()
        self.y_filter.reset()


class LandmarkSmoother:
    """
    Bộ lọc làm mượt toàn bộ 21 điểm landmark bàn tay bằng One Euro Filter.
    Mỗi landmark có một bộ lọc 2D PointFilter độc lập.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.04, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self.filters    = [
            PointFilter(min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            for _ in range(21)
        ]

    def filter_landmarks(
        self, landmark_list_px: list, timestamp: Optional[float] = None, return_int: bool = True
    ) -> list:
        """
        Làm mượt danh sách landmark (21 điểm).
        """
        if not landmark_list_px:
            return []
        if timestamp is None:
            timestamp = time.time()

        smoothed = []
        for i, pt in enumerate(landmark_list_px):
            if i >= len(self.filters):
                self.filters.append(
                    PointFilter(min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                )

            if len(pt) >= 4:
                lm_id, x, y, z = pt[0], pt[1], pt[2], pt[3]
                fx, fy = self.filters[i].filter(float(x), float(y), timestamp)
                if return_int:
                    smoothed.append((lm_id, int(round(fx)), int(round(fy)), z))
                else:
                    smoothed.append((lm_id, fx, fy, z))
            elif len(pt) == 3:
                lm_id, x, y = pt[0], pt[1], pt[2]
                fx, fy = self.filters[i].filter(float(x), float(y), timestamp)
                if return_int:
                    smoothed.append((lm_id, int(round(fx)), int(round(fy))))
                else:
                    smoothed.append((lm_id, fx, fy))
            elif len(pt) == 2:
                x, y = pt[0], pt[1]
                fx, fy = self.filters[i].filter(float(x), float(y), timestamp)
                if return_int:
                    smoothed.append((int(round(fx)), int(round(fy))))
                else:
                    smoothed.append((fx, fy))
            else:
                smoothed.append(pt)

        return smoothed

    def reset(self) -> None:
        """Reset toàn bộ 21 bộ lọc PointFilter."""
        for f in self.filters:
            f.reset()
