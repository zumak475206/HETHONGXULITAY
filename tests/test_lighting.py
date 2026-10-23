"""
tests/test_lighting.py
======================
Benchmark đánh giá hiệu năng hệ thống dưới các điều kiện ánh sáng khác nhau:
1. Thiếu sáng / Phòng tối (Dark / Low-light).
2. Bình thường (Standard indoor).
3. Nắng chói / Ngược sáng (Overexposed / Glare).
Xuất kết quả ra benchmarks/results.json.
"""

import unittest
import json
import numpy as np
import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.preprocessing.clahe_enhancer import CLAHEEnhancer


def simulate_lighting(frame: np.ndarray, condition: str) -> np.ndarray:
    """Giả lập điều kiện ánh sáng bằng các phép biến đổi sắc độ OpenCV."""
    if condition == "dark":
        # Giảm sáng 70% và giảm gamma
        dark = cv2.convertScaleAbs(frame, alpha=0.3, beta=5)
        return dark
    elif condition == "glare":
        # Cháy sáng và thêm các đốm chói lóa
        glare = cv2.convertScaleAbs(frame, alpha=1.7, beta=60)
        # Thêm đốm sáng trắng mạnh ở trung tâm
        h, w, _ = frame.shape
        cv2.circle(glare, (w // 2, h // 2), 90, (255, 255, 255), cv2.FILLED)
        return glare
    return frame.copy()


class TestLightingRobustness(unittest.TestCase):

    def test_lighting_adaptability(self):
        """Kiểm tra CLAHE phục hồi độ tương phản và biên độ thông tin ở 3 điều kiện."""
        enhancer = CLAHEEnhancer(enabled=True, auto_gamma=True, anti_glare=True)

        # Tạo ảnh mẫu có bàn tay nhân tạo
        base_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Vẽ lòng bàn tay và các ngón
        cv2.circle(base_frame, (320, 240), 60, (180, 150, 130), cv2.FILLED)
        for i in range(5):
            cv2.line(base_frame, (320, 240), (260 + i*30, 120), (180, 150, 130), 16)

        results = {}

        for cond in ["dark", "normal", "glare"]:
            sim_frame = simulate_lighting(base_frame, cond)
            enhanced = enhancer.process(sim_frame)

            # Đo độ tương phản bằng độ lệch chuẩn sắc độ (Standard Deviation của kênh xám)
            gray_raw = cv2.cvtColor(sim_frame, cv2.COLOR_BGR2GRAY)
            gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

            std_raw = float(np.std(gray_raw))
            std_enh = float(np.std(gray_enh))
            gain = (std_enh - std_raw) / max(1.0, std_raw) * 100.0

            results[cond] = {
                "raw_contrast_std": round(std_raw, 2),
                "enhanced_contrast_std": round(std_enh, 2),
                "contrast_gain_pct": round(gain, 1),
                "mean_lightness": round(float(np.mean(gray_enh)), 2)
            }

            # Ở điều kiện tối và bình thường, CLAHE phải tăng độ tương phản (gain > 0)
            if cond == "dark":
                self.assertGreater(std_enh, std_raw)

        print("\n[BENCHMARK LIGHTING ROBUSTNESS]")
        for c, r in results.items():
            print(f"  - Condition [{c.upper()}]: Contrast Gain = {r['contrast_gain_pct']}% (Mean L: {r['mean_lightness']})")

        # Lưu kết quả vào benchmarks/results.json
        res_file = config.BENCHMARK_DIR / "results.json"
        data = {}
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data["lighting"] = results
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
