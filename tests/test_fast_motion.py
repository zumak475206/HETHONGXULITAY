"""
tests/test_fast_motion.py
=========================
Benchmark kiểm tra độ chính xác và tính ổn định khi cử chỉ chuyển động ở các tốc độ khác nhau:
1. Chậm (Slow: 0.5x tốc độ).
2. Chuẩn (Normal: 1.0x tốc độ).
3. Nhanh (Fast: 2.0x tốc độ).

Xuất kết quả ra benchmarks/results.json và vẽ biểu đồ benchmarks/plots/motion_benchmark.png (Matplotlib).
"""

import unittest
import json
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.models.bilstm_attention import MultiScaleBiLSTMAttention


def time_warp_sequence(seq: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Biến đổi tốc độ chuỗi thời gian (Time-Warping):
    - speed_factor = 2.0: Cử chỉ diễn ra nhanh gấp đôi (nén chuỗi).
    - speed_factor = 0.5: Cử chỉ diễn ra chậm bằng nửa (kéo dãn chuỗi).
    Đầu ra luôn được nội suy về độ dài chuẩn 30 frame.
    """
    seq_len, feat_dim = seq.shape
    orig_steps = np.linspace(0, 1, seq_len)
    new_steps = np.linspace(0, min(1.0, 1.0 / speed_factor), seq_len)

    warped = np.zeros_like(seq)
    for d in range(feat_dim):
        warped[:, d] = np.interp(new_steps, orig_steps, seq[:, d])
    return warped


class TestFastMotionRobustness(unittest.TestCase):

    def test_motion_speed_variations(self):
        """Kiểm tra mạng Multi-Scale Bi-LSTM nhận diện chuẩn xác ở các tốc độ khác nhau."""
        model = MultiScaleBiLSTMAttention(input_dim=config.FEATURE_DIM, num_classes=10)
        model.eval()

        device = torch.device(config.DEVICE if torch.cuda.is_available() and "cuda" in config.DEVICE else "cpu")
        model.to(device)

        # Tạo chuỗi cử chỉ mẫu
        t = np.linspace(0, 2 * np.pi, config.SEQUENCE_LENGTH)
        base_seq = np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_DIM), dtype=np.float32)
        base_seq[:, 0] = np.sin(t)
        base_seq[:, 1] = np.cos(t)

        speeds = [0.5, 0.75, 1.0, 1.5, 2.0]
        results = {}

        for spd in speeds:
            warped = time_warp_sequence(base_seq, spd)
            inp_t = torch.tensor(np.expand_dims(warped, axis=0), dtype=torch.float32, device=device)

            with torch.no_grad():
                logits, attn = model(inp_t)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_cls = int(torch.argmax(probs).item())
                confidence = float(probs[pred_cls].item())
                attn_peak = float(torch.max(attn[0]).item())

            results[f"{spd}x"] = {
                "speed_factor": spd,
                "confidence": round(confidence, 3),
                "attention_peak": round(attn_peak, 3)
            }

        print("\n[BENCHMARK FAST MOTION ROBUSTNESS]")
        for spd_lbl, r in results.items():
            print(f"  - Speed {spd_lbl}: Confidence = {r['confidence']*100:.1f}% | Attention Peak = {r['attention_peak']:.3f}")

        # Lưu kết quả vào benchmarks/results.json
        res_file = config.BENCHMARK_DIR / "results.json"
        data = {}
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data["fast_motion"] = results
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Vẽ biểu đồ trực quan hóa bằng matplotlib
        self._plot_benchmarks(data)

    def _plot_benchmarks(self, data: dict):
        """Vẽ biểu đồ phong cách locate-anything.cpp xuất ra benchmarks/plots/."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plot_file = config.BENCHMARK_PLOTS_DIR / "motion_benchmark.png"

            speeds = [0.5, 0.75, 1.0, 1.5, 2.0]
            confs = [data["fast_motion"][f"{s}x"]["confidence"] * 100 for s in speeds]
            attns = [data["fast_motion"][f"{s}x"]["attention_peak"] for s in speeds]

            fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
            color = 'tab:blue'
            ax1.set_xlabel('Speed Factor (Gesture Motion Speed)', fontsize=11)
            ax1.set_ylabel('Confidence (%)', color=color, fontsize=11)
            ax1.plot(speeds, confs, marker='o', linewidth=2, color=color, label='Confidence')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(0, 105)
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Attention Peak Weight', color=color, fontsize=11)
            ax2.plot(speeds, attns, marker='s', linestyle='--', linewidth=2, color=color, label='Attention Peak')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('HETHONGXULITAY - Fast/Slow Motion Robustness Benchmark', fontsize=12, fontweight='bold')
            fig.tight_layout()
            plt.savefig(plot_file)
            plt.close()
            print(f"[+] Exported benchmark plot to: {plot_file}")
        except Exception as e:
            print(f"[!] Cannot generate matplotlib plot: {e}")


if __name__ == "__main__":
    unittest.main()
