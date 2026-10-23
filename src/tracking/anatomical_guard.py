"""
src/tracking/anatomical_guard.py
=================================
Bộ lọc giải phẫu học sinh học (Anatomical Sanity Guard).

Mục tiêu kỹ thuật:
Chặn đứng 100% hiện tượng ảo giác hồi quy (Landmark Hallucination / Skin-Blending)
khiến các điểm MediaPipe bị kéo dãn bất thường lên bờ vai, ngực hoặc khuôn mặt
khi bàn tay tiếp xúc gần da cơ thể.

Quy luật giải phẫu bất biến của bàn tay người (Biometric Invariants):
1. Tỷ lệ ngón tay so với kích thước lòng bàn tay (Finger-to-Palm Ratio):
   Chiều dài ngón tay (L_finger) không bao giờ vượt quá 1.25 lần kích thước lòng bàn tay (L_palm):
   L_finger / L_palm <= 1.25.
2. Tỷ lệ các đốt xương (Phalanx Proportion - Golden Ratio):
   - Đốt gần (Proximal) / Đốt giữa (Intermediate) xấp xỉ tỷ lệ vàng 1.618.
   - Không có đốt xương đơn lẻ nào vượt quá 0.85 lần kích thước lòng bàn tay.
   - Khi landmark bị kéo lên vai/ngực, chiều dài đốt xương bị kéo dãn bất thường (gấp 3-5 lần),
     làm tỷ lệ đốt xương biến dạng đột ngột.
3. Kích thước tối thiểu của lòng bàn tay (Minimum Palm Size):
   Tránh chia cho 0 và loại bỏ các hộp nhận diện giả quá nhỏ.
"""

import math
from typing import List, Tuple, Optional, Any, Union
import numpy as np


class AnatomicalGuard:
    """
    Bộ lọc kiểm tra tính hợp lệ sinh học của bộ khung xương bàn tay (21 landmarks).
    """

    def __init__(
        self,
        enabled: bool = True,
        max_finger_palm_ratio: float = 1.25,
        max_segment_palm_ratio: float = 0.85,
        min_phalanx_ratio: float = 0.35,
        max_phalanx_ratio: float = 3.50,
        min_palm_size: float = 10.0,
    ):
        """
        Tham số
        -------
        enabled : bool
            Bật/tắt bộ lọc bảo vệ.
        max_finger_palm_ratio : float
            Tỷ lệ chiều dài tối đa của ngón tay so với lòng bàn tay (mặc định 1.25).
        max_segment_palm_ratio : float
            Tỷ lệ tối đa của 1 đốt xương đơn lẻ so với lòng bàn tay (mặc định 0.85).
        min_phalanx_ratio : float
            Tỷ lệ tối thiểu giữa đốt gần và đốt giữa (mặc định 0.35).
        max_phalanx_ratio : float
            Tỷ lệ tối đa giữa đốt gần và đốt giữa (mặc định 3.50).
        min_palm_size : float
            Kích thước lòng bàn tay tối thiểu chấp nhận được tính theo pixel (mặc định 10.0).
        """
        self.enabled = bool(enabled)
        self.max_finger_palm_ratio = float(max_finger_palm_ratio)
        self.max_segment_palm_ratio = float(max_segment_palm_ratio)
        self.min_phalanx_ratio = float(min_phalanx_ratio)
        self.max_phalanx_ratio = float(max_phalanx_ratio)
        self.min_palm_size = float(min_palm_size)

        # Định nghĩa các chỉ số landmark của từng ngón tay
        # [MCP, PIP, DIP, TIP]
        self.FINGERS = {
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }
        # Ngón cái: [CMC, MCP, IP, TIP]
        self.THUMB = (1, 2, 3, 4)

    @staticmethod
    def _extract_points(lm_list: Any) -> Optional[np.ndarray]:
        """
        Trích xuất mảng tọa độ (21, 2) từ các định dạng đầu vào khác nhau:
        - List các tuple [(id, x, y, z), ...] hoặc [(id, x, y), ...]
        - List các điểm [(x, y), ...]
        - MediaPipe NormalizedLandmarkList (object có thuộc tính .landmark)
        """
        if lm_list is None:
            return None

        # Trường hợp MediaPipe NormalizedLandmarkList
        if hasattr(lm_list, "landmark"):
            lm_list = lm_list.landmark

        if len(lm_list) < 21:
            return None

        pts = np.zeros((21, 2), dtype=np.float64)
        try:
            for i in range(21):
                pt = lm_list[i]
                if hasattr(pt, "x") and hasattr(pt, "y"):
                    pts[i, 0] = float(pt.x)
                    pts[i, 1] = float(pt.y)
                elif isinstance(pt, dict):
                    pts[i, 0] = float(pt["x"])
                    pts[i, 1] = float(pt["y"])
                elif len(pt) >= 4:
                    if isinstance(pt[0], (int, np.integer)) and int(pt[0]) == i:
                        pts[i, 0] = float(pt[1])
                        pts[i, 1] = float(pt[2])
                    else:
                        pts[i, 0] = float(pt[0])
                        pts[i, 1] = float(pt[1])
                elif len(pt) == 3:
                    if isinstance(pt[0], (int, np.integer)) and int(pt[0]) == i:
                        pts[i, 0] = float(pt[1])
                        pts[i, 1] = float(pt[2])
                    else:
                        pts[i, 0] = float(pt[0])
                        pts[i, 1] = float(pt[1])
                elif len(pt) == 2:
                    pts[i, 0] = float(pt[0])
                    pts[i, 1] = float(pt[1])
                else:
                    return None
        except Exception:
            return None

        # Kiểm tra NaN hoặc Inf
        if not np.all(np.isfinite(pts)):
            return None

        return pts

    def validate_hand(
        self, lm_list: Any, palm_size: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ sinh học giải phẫu của bàn tay.

        Tham số
        -------
        lm_list : Any
            Danh sách 21 điểm landmark bàn tay.
        palm_size : Optional[float]
            Kích thước lòng bàn tay đã tính trước (nếu có).

        Trả về
        ------
        (is_valid: bool, reason: str)
            is_valid : True nếu khung xương hợp lệ, False nếu phát hiện dị dạng.
            reason   : Chuỗi giải thích lý do (thành công hoặc vi phạm cụ thể).
        """
        if not self.enabled:
            return True, "AnatomicalGuard disabled"

        pts = self._extract_points(lm_list)
        if pts is None:
            return False, "Invalid landmark format or non-finite coordinates"

        def dist(i: int, j: int) -> float:
            return math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])

        # ── 1. Kiểm tra hình học khung lòng bàn tay (Palm Framework Consistency) ───
        length_wrist_middle = dist(0, 9)   # WRIST -> MIDDLE_MCP
        knuckle_width       = dist(5, 17)  # INDEX_MCP -> PINKY_MCP

        # Ngưỡng kích thước tối thiểu (hỗ trợ cả tọa độ pixel lẫn normalized [0, 1])
        min_size = self.min_palm_size
        if np.max(pts) <= 1.05 and min_size >= 1.0:
            min_size = 0.02  # Normalized coordinate threshold

        if knuckle_width < min_size * 0.40:
            return False, f"Knuckle span too small ({knuckle_width:.2f} < {min_size * 0.40:.2f})"

        # 1a. Tỷ lệ khuôn lòng bàn tay (Palm Aspect Ratio = Chiều dài / Chiều rộng)
        # Bàn tay người có tỷ lệ xấp xỉ 0.8 - 1.5. Khi cổ tay bị kéo dãn lên vai/ngực,
        # tỷ lệ này tăng vọt (> 2.20). Khi bàn tay bị bẹp méo, tỷ lệ tụt dốc (< 0.35).
        palm_aspect_ratio = length_wrist_middle / max(knuckle_width, 1e-6)
        if palm_aspect_ratio > 2.20:
            return False, (
                f"Palm aspect ratio {palm_aspect_ratio:.2f} exceeds 2.20 "
                f"(Wrist or MCP hallucination stretching to shoulder/body)"
            )
        if palm_aspect_ratio < 0.35:
            return False, f"Palm aspect ratio {palm_aspect_ratio:.2f} below minimum 0.35"

        # 1b. Tỷ lệ chiều dài xương ngón giữa so với lòng bàn tay
        # Tổng độ dài 3 đốt ngón giữa không thể nhỏ hơn 48% chiều dài lòng bàn tay
        total_middle_len = dist(9, 10) + dist(10, 11) + dist(11, 12)
        if total_middle_len / max(length_wrist_middle, 1e-6) < 0.48:
            return False, (
                f"Middle finger bone to palm ratio {total_middle_len / length_wrist_middle:.2f} "
                f"below 0.48 (Anomalous palm elongation)"
            )

        # 1c. Kích thước lòng bàn tay chuẩn hóa bất biến (Invariant Palm Size)
        if palm_size is None or palm_size <= 0.0:
            palm_size = max(knuckle_width * 1.20, min(length_wrist_middle, knuckle_width * 1.80))

        if palm_size < min_size:
            return False, f"Palm size too small ({palm_size:.2f} < {min_size})"

        # ── 2. Kiểm tra tỷ lệ chiều dài 4 ngón tay (Trỏ, Giữa, Áp út, Út) ──────
        for name, (mcp, pip, dip, tip) in self.FINGERS.items():
            # Khoảng cách đường thẳng từ gốc MCP tới đầu ngón TIP
            finger_len_direct = dist(mcp, tip)
            direct_ratio = finger_len_direct / palm_size

            if direct_ratio > self.max_finger_palm_ratio:
                return False, (
                    f"Finger '{name}' length ratio {direct_ratio:.2f} "
                    f"exceeds max {self.max_finger_palm_ratio:.2f}"
                )

            # Chiều dài từng đốt xương
            l_proximal     = dist(mcp, pip)
            l_intermediate = dist(pip, dip)
            l_distal       = dist(dip, tip)

            # Tổng độ dài các đốt xương dọc theo cung ngón tay
            total_finger_len = l_proximal + l_intermediate + l_distal
            total_ratio = total_finger_len / palm_size
            if total_ratio > (self.max_finger_palm_ratio + 0.15):
                return False, (
                    f"Finger '{name}' total bone arc ratio {total_ratio:.2f} "
                    f"exceeds limit {self.max_finger_palm_ratio + 0.15:.2f}"
                )

            # ── 3. Kiểm tra tỷ lệ từng đốt xương đơn lẻ ────────────────────────
            max_seg = max(l_proximal, l_intermediate, l_distal)
            max_seg_ratio = max_seg / palm_size
            if max_seg_ratio > self.max_segment_palm_ratio:
                return False, (
                    f"Finger '{name}' segment ratio {max_seg_ratio:.2f} "
                    f"exceeds max {self.max_segment_palm_ratio:.2f}"
                )

            # ── 4. Kiểm tra tỷ lệ giữa các đốt ngón tay (Golden Ratio Proportions) ─
            # Chỉ kiểm tra khi đốt giữa đủ rõ ràng (> 8% palm_size) để tránh nhiễu khi ngón gập sâu
            if l_intermediate > palm_size * 0.08:
                ratio_pi = l_proximal / (l_intermediate + 1e-6)
                if ratio_pi < self.min_phalanx_ratio or ratio_pi > self.max_phalanx_ratio:
                    return False, (
                        f"Finger '{name}' proximal/intermediate ratio {ratio_pi:.2f} "
                        f"out of anatomical range [{self.min_phalanx_ratio:.2f}, {self.max_phalanx_ratio:.2f}]"
                    )

            if l_distal > palm_size * 0.08:
                ratio_id = l_intermediate / (l_distal + 1e-6)
                if ratio_id < self.min_phalanx_ratio or ratio_id > self.max_phalanx_ratio:
                    return False, (
                        f"Finger '{name}' intermediate/distal ratio {ratio_id:.2f} "
                        f"out of anatomical range [{self.min_phalanx_ratio:.2f}, {self.max_phalanx_ratio:.2f}]"
                    )

        # ── 5. Kiểm tra ngón cái ──────────────────────────────────────────────
        cmc, mcp, ip, tip = self.THUMB
        thumb_len_direct = dist(mcp, tip)
        if (thumb_len_direct / palm_size) > (self.max_finger_palm_ratio - 0.10):
            return False, (
                f"Thumb direct length ratio {thumb_len_direct / palm_size:.2f} "
                f"exceeds limit {self.max_finger_palm_ratio - 0.10:.2f}"
            )

        total_thumb_len = dist(cmc, mcp) + dist(mcp, ip) + dist(ip, tip)
        if (total_thumb_len / palm_size) > (self.max_finger_palm_ratio + 0.05):
            return False, (
                f"Thumb total arc ratio {total_thumb_len / palm_size:.2f} exceeds limit"
            )

        # ── 6. Kiểm tra khoảng cách cổ tay tới các khớp gốc (MCP) ─────────────
        min_wrist_mcp = min(dist(0, mcp_id) for mcp_id in [5, 9, 13, 17])
        if (min_wrist_mcp / max(knuckle_width, 1e-6)) > 2.00:
            return False, (
                f"Wrist detached from knuckle span (ratio {min_wrist_mcp / knuckle_width:.2f} > 2.00)"
            )

        for mcp_id in [1, 5, 9, 13, 17]:
            d_wrist_mcp = dist(0, mcp_id)
            if (d_wrist_mcp / palm_size) > 1.35:
                return False, (
                    f"Wrist to MCP({mcp_id}) distance ratio {d_wrist_mcp / palm_size:.2f} "
                    f"exceeds palm boundary limit 1.35"
                )

        return True, "Valid anatomical hand"

    def is_valid_hand(self, lm_list: Any) -> bool:
        """Hàm tiện ích trả về boolean đơn giản."""
        valid, _ = self.validate_hand(lm_list)
        return valid
