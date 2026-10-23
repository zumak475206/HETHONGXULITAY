"""
tests/test_clahe.py
===================
Kiểm thử tự động tính đúng đắn của Bộ tiền xử lý CLAHE.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.clahe_enhancer import CLAHEEnhancer


class TestCLAHEEnhancer(unittest.TestCase):

    def setUp(self):
        self.enhancer = CLAHEEnhancer(enabled=True, clip_limit=2.0, auto_gamma=True, anti_glare=True)

    def test_output_shape_and_type(self):
        """Kiểm tra ảnh sau khi xử lý giữ nguyên kích thước và định dạng uint8."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        enhanced = self.enhancer.process(frame)
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_dark_image_enhancement(self):
        """Kiểm tra ảnh tối được tăng độ tương phản và độ sáng trung bình."""
        dark_frame = np.full((100, 100, 3), fill_value=30, dtype=np.uint8)
        enhanced = self.enhancer.process(dark_frame)
        self.assertGreaterEqual(np.mean(enhanced), np.mean(dark_frame))

    def test_anti_glare_soft_clamping(self):
        """Kiểm tra vùng cháy sáng (glare > 225) thực sự được nén mềm an toàn (L <= 242)."""
        glare_frame = np.full((100, 100, 3), fill_value=250, dtype=np.uint8)
        enhanced = self.enhancer.process(glare_frame)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        # Giá trị độ sáng L phải được nén xuống <= 242 thay vì giữ nguyên 250 hoặc bị đội lên 255
        self.assertLessEqual(int(np.max(lab[:, :, 0])), 242)

    def test_edge_sharpening_option(self):
        """Kiểm tra cờ cấu hình làm sắc nét đường biên (edge_sharpen)."""
        enhancer_sharpen = CLAHEEnhancer(enabled=True, edge_sharpen=True)
        enhancer_no_sharpen = CLAHEEnhancer(enabled=True, edge_sharpen=False)
        frame = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        out_sharpen = enhancer_sharpen.process(frame)
        out_no_sharpen = enhancer_no_sharpen.process(frame)
        self.assertEqual(out_sharpen.shape, frame.shape)
        self.assertEqual(out_no_sharpen.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
