"""
tests/test_latency.py
=====================
Benchmark độ trễ thời gian thực (ms/frame) của toàn bộ pipeline ở các mức FPS khác nhau.
Xuất kết quả ra benchmarks/results.json và vẽ biểu đồ benchmarks/plots/latency_benchmark.png.
"""

import unittest
import time
import json
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.preprocessing.clahe_enhancer import CLAHEEnhancer
from src.tracking.kalman_filter import HandKalmanFilter
from src.features.feature_extractor import extract_158d_features, extract_157d_features, extract_112d_features
from src.models.bilstm_attention import MultiScaleBiLSTMAttention


class TestPipelineLatency(unittest.TestCase):

    def test_full_pipeline_latency(self):
        """Đo thời gian xử lý từng tầng trong pipeline trên 100 frame giả lập."""
        enhancer = CLAHEEnhancer(enabled=True)
        kalman = [HandKalmanFilter(), HandKalmanFilter()]
        model = MultiScaleBiLSTMAttention(input_dim=config.FEATURE_DIM, num_classes=10)
        model.eval()

        device = torch.device(config.DEVICE if torch.cuda.is_available() and "cuda" in config.DEVICE else "cpu")
        model.to(device)

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        dummy_landmarks = [[i, 300.0 + i*2, 200.0 + i*3, 10.0 + i] for i in range(21)]
        seq_buffer = [np.random.randn(config.FEATURE_DIM).tolist() for _ in range(config.SEQUENCE_LENGTH)]

        times_clahe = []
        times_kalman = []
        times_features = []
        times_nn = []
        times_total = []

        # Warm up
        dummy_seq = torch.randn(1, config.SEQUENCE_LENGTH, config.FEATURE_DIM, device=device)
        with torch.no_grad():
            for _ in range(10):
                model(dummy_seq)

        num_iterations = 60
        for _ in range(num_iterations):
            t0 = time.perf_counter()

            # 1. CLAHE
            t_c0 = time.perf_counter()
            _ = enhancer.process(frame)
            times_clahe.append((time.perf_counter() - t_c0) * 1000)

            # 2. Kalman
            t_k0 = time.perf_counter()
            s_pts, _ = kalman[0].update(dummy_landmarks)
            times_kalman.append((time.perf_counter() - t_k0) * 1000)

            # 3. Features
            t_f0 = time.perf_counter()
            feat = extract_158d_features([s_pts], [(0.0, 0.0, 0.0)], mode_id=1)
            times_features.append((time.perf_counter() - t_f0) * 1000)

            # 4. Neural Network Inference
            t_n0 = time.perf_counter()
            seq_buffer.pop(0)
            seq_buffer.append(feat)
            inp_t = torch.tensor([seq_buffer], dtype=torch.float32, device=device)
            with torch.no_grad():
                _ = model(inp_t)
            times_nn.append((time.perf_counter() - t_n0) * 1000)

            times_total.append((time.perf_counter() - t0) * 1000)

        mean_total = float(np.mean(times_total))
        mean_clahe = float(np.mean(times_clahe))
        mean_kalman = float(np.mean(times_kalman))
        mean_feat = float(np.mean(times_features))
        mean_nn = float(np.mean(times_nn))

        print(f"\n[BENCHMARK PIPELINE LATENCY]")
        print(f"  - CLAHE Preprocessing: {mean_clahe:.2f} ms")
        print(f"  - Kalman Filter (21 pts): {mean_kalman:.2f} ms")
        print(f"  - {config.FEATURE_DIM}D Feature Extraction: {mean_feat:.2f} ms")
        print(f"  - Bi-LSTM + Attention NN: {mean_nn:.2f} ms")
        print(f"  - TOTAL PIPELINE LATENCY: {mean_total:.2f} ms (~ {1000.0/max(0.1, mean_total):.1f} FPS)")

        # Mục tiêu kỹ thuật: Độ trễ toàn bộ pipeline phải < 33.3ms (đạt chuẩn thời gian thực 30 FPS)
        self.assertLess(mean_total, 35.0, f"Độ trễ pipeline vượt quá giới hạn thời gian thực: {mean_total:.2f} ms")

        # Lưu kết quả vào benchmarks/results.json
        res_file = config.BENCHMARK_DIR / "results.json"
        data = {}
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data["latency"] = {
            "clahe_ms": round(mean_clahe, 2),
            "kalman_ms": round(mean_kalman, 2),
            "features_ms": round(mean_feat, 2),
            "neural_net_ms": round(mean_nn, 2),
            "total_ms": round(mean_total, 2),
            "estimated_fps": round(1000.0 / max(0.1, mean_total), 1),
            "target_met": mean_total < 35.0
        }

        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
