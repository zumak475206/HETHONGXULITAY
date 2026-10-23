"""
src/preprocessing/clahe_enhancer.py
===================================
Bộ tiền xử lý thị giác máy tính tăng cường tương phản thích ứng cục bộ (CLAHE).

Bản chất kỹ thuật:
1. Không gian màu LAB:
   - Tách biệt hoàn toàn thông tin độ sáng (L - Lightness [0, 255]) khỏi thông tin màu sắc (A, B).
   - Tăng tương phản trên kênh L không làm biến dạng hoặc đổi màu da của bàn tay.
2. Thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization):
   - Chia ảnh thành lưới các ô nhỏ (Grid, mặc định 8x8).
   - Cân bằng biểu đồ sắc độ (Histogram Equalization) cục bộ trong từng ô.
   - Áp dụng ngưỡng cắt (Clip Limit = 2.0) để hạn chế khuếch đại hạt nhiễu (noise spike) ở vùng tối.
   - Nội suy song tuyến tính (Bilinear Interpolation) giữa các ranh giới ô để triệt tiêu đường viền ghép.
3. Chống chói lóa (Anti-Glare Soft-Clamping):
   - Nén các điểm ảnh bị cháy sáng (L > 225) theo hàm phi tuyến để giữ lại chi tiết ranh giới ngón tay.
4. Dynamic Gamma:
   - Đo độ sáng trung bình để tự động hiệu chỉnh Gamma: tăng sáng khi phòng tối, giảm sáng khi phòng chói.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CLAHEEnhancer:
    """
    Lớp xử lý tăng cường hình ảnh thích ứng cho camera thị giác máy tính.
    """

    def __init__(
        self,
        enabled: bool = True,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
        auto_gamma: bool = True,
        anti_glare: bool = True,
        edge_sharpen: bool = True,
    ):
        self.enabled = enabled
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.auto_gamma = auto_gamma
        self.anti_glare = anti_glare
        self.edge_sharpen = edge_sharpen

        # Khởi tạo đối tượng CLAHE của OpenCV
        self._clahe = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=tuple(self.grid_size)
        )

        # Bảng tra cứu 256-LUT nén mềm vùng cháy sáng (Highlight Soft-Clamping > 225)
        self._anti_glare_lut = self._build_anti_glare_lut()

        # Bộ nhớ đệm bảng tra cứu (Lookup Table - LUT) để tối ưu tốc độ (<0.1ms)
        self._current_gamma = 1.0
        self._gamma_lut = self._build_gamma_lut(1.0)
        self._frame_count = 0
        self._last_mean_l = 128.0

    def _build_anti_glare_lut(self) -> np.ndarray:
        """Tạo bảng tra cứu 256 phần tử cho nén mềm vùng chói sáng (Highlight Soft-Clamping > 225)."""
        table = np.arange(256, dtype=np.uint8)
        for i in range(226, 256):
            table[i] = min(255, int(225 + (i - 225) * 0.4 + 0.5))
        return table

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        """Tạo bảng tra cứu 256 phần tử cho biến đổi Gamma phi tuyến: O = 255 * (I/255)^(1/gamma)."""
        inv_gamma = 1.0 / max(0.1, gamma)
        table = np.array([
            min(255, int(((i / 255.0) ** inv_gamma) * 255 + 0.5))
            for i in range(256)
        ], dtype=np.uint8)
        return table

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Nhận ảnh BGR từ webcam và trả về ảnh BGR đã tăng cường tương phản và khử nhiễu/chói.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return frame

        self._frame_count += 1

        # 1. Chuyển đổi sang không gian màu LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Dynamic Gamma Adaptation (Cập nhật mỗi 10 frame để tiết kiệm CPU)
        if self.auto_gamma and (self._frame_count % 10 == 1):
            self._last_mean_l = float(np.mean(l_channel))
            # Nếu tối (< 85) -> gamma 1.2 - 1.5 để làm sáng chi tiết
            # Nếu quá sáng (> 170) -> gamma 0.7 - 0.9 để giảm chói
            if self._last_mean_l < 85.0:
                target_gamma = 1.0 + (85.0 - self._last_mean_l) / 85.0 * 0.5
            elif self._last_mean_l > 170.0:
                target_gamma = 1.0 - (self._last_mean_l - 170.0) / 85.0 * 0.35
            else:
                target_gamma = 1.0

            target_gamma = float(np.clip(target_gamma, 0.65, 1.50))
            if abs(target_gamma - self._current_gamma) > 0.05:
                self._current_gamma = target_gamma
                self._gamma_lut = self._build_gamma_lut(self._current_gamma)

        # Áp dụng Gamma qua bảng LUT siêu tốc
        if self.auto_gamma and abs(self._current_gamma - 1.0) > 0.02:
            l_channel = cv2.LUT(l_channel, self._gamma_lut)

        # 3. Áp dụng CLAHE cân bằng độ tương phản cục bộ trên kênh L
        l_enhanced = self._clahe.apply(l_channel)

        # 4. Làm rõ biên cạnh da tay (Local Contrast Edge Sharpening qua Unsharp Masking)
        if self.edge_sharpen:
            blurred_l = cv2.GaussianBlur(l_enhanced, (0, 0), sigmaX=1.5)
            l_enhanced = cv2.addWeighted(l_enhanced, 1.25, blurred_l, -0.25, 0)

        # 5. Chống chói lóa (256-LUT Highlight Soft-Clamping vùng quá sáng > 225)
        # BẮT BUỘC áp dụng SAU CLAHE và Sharpening để triệt tiêu triệt để các đốm cháy sáng phát sinh
        if self.anti_glare:
            l_enhanced = cv2.LUT(l_enhanced, self._anti_glare_lut)

        # 6. Gộp các kênh lại và chuyển ngược về BGR
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def get_lightness_stats(self) -> dict:
        """Trả về thông số độ sáng hiện tại để hiển thị HUD chẩn đoán môi trường."""
        return {
            "mean_l": self._last_mean_l,
            "current_gamma": self._current_gamma,
            "status": "Tối" if self._last_mean_l < 85 else ("Chói" if self._last_mean_l > 170 else "Chuẩn")
        }
