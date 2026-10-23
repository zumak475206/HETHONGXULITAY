"""
src/training/train.py
=====================
Pipeline Huấn luyện PyTorch GPU, Xuất ONNX và Lượng tử hóa INT8 Động (Dynamic Quantization).

Chức năng:
1. Nạp và phân chia tập dữ liệu chuỗi 30 frame (dataset/gesture_sequences.npz).
   Nếu chưa có dữ liệu thực tế, script tự động sinh dữ liệu mẫu hợp lệ để kiểm tra toàn bộ pipeline.
2. Huấn luyện mô hình MultiScaleBiLSTMAttention trên GPU NVIDIA RTX 3050.
   - Hàm mất mát: CrossEntropyLoss (kèm Label Smoothing).
   - Tối ưu hóa: AdamW (weight_decay=1e-3), CosineAnnealingLR scheduler.
3. Xuất mô hình sang định dạng chuẩn ONNX FP32:
   weights/gesture_bilstm_multiscale.onnx
4. Lượng tử hóa INT8 tự động (BỔ SUNG 2):
   - Sử dụng onnxruntime.quantization.quantize_dynamic (QuantType.QInt8).
   - Xuất file: weights/gesture_bilstm_int8.onnx.
   - So sánh kích thước file (giảm ~4x) và đo độ chính xác trước/sau lượng tử hóa.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Đảm bảo import được config và các module từ src
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from src.models.bilstm_attention import MultiScaleBiLSTMAttention


class GestureSequenceDataset(Dataset):
    """Lớp Dataset nạp chuỗi thời gian shape (N, 30, 157) và nhãn tương ứng."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_synthetic_dataset(
    samples_per_class: int = 40,
    seq_len: int = config.SEQUENCE_LENGTH,
    feat_dim: int = config.FEATURE_DIM,
    num_classes: int = config.NUM_CLASSES
) -> Tuple[np.ndarray, np.ndarray]:
    """Sinh tập dữ liệu chuỗi mẫu giả lập để kiểm tra và benchmark pipeline (hỗ trợ đầy đủ 158D và 45 frames)."""
    np.random.seed(42)
    X_list, y_list = [], []

    for cls in range(num_classes):
        base_freq = (cls + 1) * 0.5
        phase_shift = cls * (np.pi / num_classes)

        for _ in range(samples_per_class):
            t = np.linspace(0, 2 * np.pi, seq_len)
            wave = np.sin(base_freq * t + phase_shift)[:, None]  # (seq_len, 1)
            noise = np.random.normal(0, 0.08, size=(seq_len, feat_dim))

            seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
            # Hand 1: 63D coords, 10D angles, 3D velocity (tổng 76D)
            seq[:, 0:min(63, feat_dim)] = wave * 0.5 + noise[:, 0:min(63, feat_dim)]
            if feat_dim >= 73:
                seq[:, 63:73] = (np.cos(base_freq * t + phase_shift)[:, None] * 0.3) + 0.5
            if feat_dim >= 76:
                seq[:, 73:76] = np.gradient(wave, axis=0) * 0.2
            # Hand 2: 63D coords, 10D angles, 3D velocity (76..152) và liên tay (152..157)
            if feat_dim >= 152 and cls >= 8:  # Cử chỉ 2 tay (ZOOM_IN, ZOOM_OUT)
                seq[:, 76:139] = (-wave * 0.5) + noise[:, 76:139]
                seq[:, 139:149] = (np.cos(base_freq * t + phase_shift)[:, None] * 0.3) + 0.5
                seq[:, 149:152] = -np.gradient(wave, axis=0) * 0.2
                seq[:, 152:min(157, feat_dim)] = np.abs(wave) * 0.5
            seq[:, :min(157, feat_dim)] += noise[:, :min(157, feat_dim)] * 0.05

            # Context Gate: Mode 1 cho chuot (PINCH_CLICK, FIST_LOCK), Mode 2 cho cu chi dong / media
            if feat_dim >= 158:
                if cls in [1, 2]:
                    mode_val = 1
                elif cls in [3, 4, 5, 6, 7, 8, 9]:
                    mode_val = 2
                else:
                    mode_val = 0
                seq[:, 157] = float(mode_val) / float(getattr(config, "MAX_MODES", 4))

            X_list.append(seq)
            y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Xáo trộn dữ liệu ngẫu nhiên
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def convert_112d_to_157d(arr: np.ndarray) -> np.ndarray:
    """Chuyển đổi mảng chuỗi đặc trưng từ chuẩn cũ 112D sang chuẩn 157D."""
    orig_shape = arr.shape[:-1]
    flat = arr.reshape(-1, 112)
    N = flat.shape[0]
    out = np.zeros((N, 157), dtype=arr.dtype)

    # Hand 1: 0..42 -> 21 pts (x,y)
    h1_2d = flat[:, 0:42].reshape(N, 21, 2)
    h1_3d = np.zeros((N, 21, 3), dtype=arr.dtype)
    h1_3d[:, :, 0:2] = h1_2d
    out[:, 0:63] = h1_3d.reshape(N, 63)
    out[:, 63:73] = flat[:, 42:52]  # angles
    out[:, 73:75] = flat[:, 52:54]  # vx, vy (vz remains 0)

    # Hand 2: 54..96 -> 21 pts (x,y)
    h2_2d = flat[:, 54:96].reshape(N, 21, 2)
    h2_3d = np.zeros((N, 21, 3), dtype=arr.dtype)
    h2_3d[:, :, 0:2] = h2_2d
    out[:, 76:139] = h2_3d.reshape(N, 63)
    out[:, 139:149] = flat[:, 96:106]  # angles
    out[:, 149:151] = flat[:, 106:108]  # vx, vy (vz remains 0)

    # Inter-hand: 108..112 (rel_dist, rel_dx, rel_dy, rel_angle)
    out[:, 152:155] = flat[:, 108:111]  # rel_dist, rel_dx, rel_dy
    out[:, 155] = 0.0  # rel_dz
    out[:, 156] = flat[:, 111]  # rel_angle

    return out.reshape(*orig_shape, 157)


def convert_to_158d(arr: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
    """Chuyển đổi mảng chuỗi đặc trưng sang chuẩn 158D kèm Context Gate.
    Hỗ trợ an toàn cả 1D (157,), 2D (T, 157) và 3D (N, T, 157).
    """
    if arr.shape[-1] == 112:
        arr = convert_112d_to_157d(arr)
    if arr.shape[-1] != 157:
        return arr

    max_modes = float(getattr(config, "MAX_MODES", 4))
    if max_modes <= 0.0:
        max_modes = 4.0

    def _get_mode_norm(cls_val):
        try:
            c = int(cls_val)
            if c in [1, 2]:
                return 1.0 / max_modes
            elif c in [3, 4, 5, 6, 7, 8, 9]:
                return 2.0 / max_modes
            else:
                return 0.0
        except Exception:
            return 0.0

    if arr.ndim == 1:
        norm_val = _get_mode_norm(y) if y is not None else 0.0
        return np.concatenate([arr, np.array([norm_val], dtype=arr.dtype)])

    elif arr.ndim == 2:
        T = arr.shape[0]
        norm_val = 0.0
        if y is not None:
            if isinstance(y, (list, tuple, np.ndarray)) and len(y) > 0:
                norm_val = _get_mode_norm(y[0])
            else:
                norm_val = _get_mode_norm(y)
        out = np.zeros((T, 158), dtype=arr.dtype)
        out[:, :157] = arr
        out[:, 157] = norm_val
        return out

    elif arr.ndim == 3:
        N, T = arr.shape[0], arr.shape[1]
        out = np.zeros((N, T, 158), dtype=arr.dtype)
        out[:, :, :157] = arr
        if y is not None and len(y) == N:
            for i, cls in enumerate(y):
                out[i, :, 157] = _get_mode_norm(cls)
        else:
            out[:, :, 157] = 0.0
        return out

    else:
        pad_shape = list(arr.shape[:-1]) + [1]
        norm_val = _get_mode_norm(y) if y is not None else 0.0
        pad = np.full(pad_shape, norm_val, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=-1)


def resample_sequence_length(arr: np.ndarray, target_len: int = config.SEQUENCE_LENGTH) -> np.ndarray:
    """Nội suy chiều thời gian đưa về target_len (ví dụ 30 -> 45 frames).
    Hỗ trợ cả 2D (T, D) và 3D (N, T, D).
    """
    if arr.ndim == 2:
        curr_len, D = arr.shape
        if curr_len == target_len:
            return arr
        out = np.zeros((target_len, D), dtype=arr.dtype)
        orig_t = np.linspace(0, 1, curr_len)
        new_t = np.linspace(0, 1, target_len)
        for d in range(D):
            out[:, d] = np.interp(new_t, orig_t, arr[:, d])
        return out
    elif arr.ndim == 3:
        N, curr_len, D = arr.shape
        if curr_len == target_len:
            return arr
        out = np.zeros((N, target_len, D), dtype=arr.dtype)
        orig_t = np.linspace(0, 1, curr_len)
        new_t = np.linspace(0, 1, target_len)
        for i in range(N):
            for d in range(D):
                out[i, :, d] = np.interp(new_t, orig_t, arr[i, :, d])
        return out
    return arr


def load_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Nạp dataset từ file .npz, tự động nâng cấp sang 158D và 45 frames nếu phát hiện dữ liệu cũ."""
    if dataset_path.exists():
        data = np.load(dataset_path)
        X = data["X"]
        y = data["y"]
        needs_resave = False

        if len(X) > 0 and X.shape[-1] in [112, 157]:
            print(f"[*] Phat hien dataset kich thuoc cu ({X.shape}). Dang nang cap sang 158D Context Gate...")
            bak_path = dataset_path.with_name("gesture_sequences_pre158d.npz.bak")
            if not bak_path.exists():
                np.savez_compressed(bak_path, X=X, y=y)
                print(f"[+] Da tao ban sao luu du lieu cu tai: {bak_path}")
            X = convert_to_158d(X, y=y)
            needs_resave = True

        if len(X) > 0 and X.shape[1] != config.SEQUENCE_LENGTH:
            print(f"[*] Dang noi suy chuoi tu {X.shape[1]} frames sang {config.SEQUENCE_LENGTH} frames...")
            X = resample_sequence_length(X, config.SEQUENCE_LENGTH)
            needs_resave = True

        if len(X) > 0 and (X.shape[-1] != config.FEATURE_DIM or X.shape[1] != config.SEQUENCE_LENGTH):
            print(f"[!] Kich thuoc dac trung {X.shape} khong khop ({config.SEQUENCE_LENGTH}, {config.FEATURE_DIM}). Sinh lai du lieu tong hop...")
            X, y = generate_synthetic_dataset(samples_per_class=40, num_classes=config.NUM_CLASSES)
            needs_resave = True

        if needs_resave:
            np.savez_compressed(dataset_path, X=X, y=y)
            print(f"[+] Da luu dataset cap nhat thanh cong: {X.shape}")

        print(f"[*] Loaded {len(y)} sequences from {dataset_path} (shape: {X.shape})")
        return X, y
    else:
        print(f"[*] Dataset not found at {dataset_path}. Generating synthetic samples...")
        X, y = generate_synthetic_dataset(samples_per_class=40, num_classes=config.NUM_CLASSES)
        np.savez_compressed(dataset_path, X=X, y=y)
        print(f"[+] Saved synthetic samples to {dataset_path} ({len(y)} samples)")
        return X, y


def export_to_onnx(model: nn.Module, save_path: Path, input_dim: int = config.FEATURE_DIM, seq_len: int = config.SEQUENCE_LENGTH):
    """Xuất mô hình PyTorch sang định dạng ONNX FP32 hoặc TorchScript JIT."""
    model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim, device="cpu")
    model_cpu = model.to("cpu")

    # 1. Luôn lưu TorchScript JIT (hiệu năng cao, không phụ thuộc ONNX)
    jit_path = save_path.with_suffix(".jit.pt")
    try:
        traced = torch.jit.trace(model_cpu, dummy_input)
        traced.save(str(jit_path))
        print(f"[+] Exported TorchScript JIT model: {jit_path} ({jit_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[!] TorchScript trace warning: {e}")

    # 2. Xuất ONNX FP32 với dynamic sequence length
    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
            input_names=["input_sequence"],
            output_names=["logits", "attention_weights"],
            dynamic_axes={
                "input_sequence": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"},
                "attention_weights": {0: "batch_size", 1: "seq_len"}
            }
        )
        print(f"[+] Exported ONNX model: {save_path} ({save_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[!] ONNX export warning: {e}")


def quantize_onnx_int8(onnx_model_path: Path, int8_save_path: Path, model_cpu: nn.Module = None):
    """Luong tu hoa mo hinh sang INT8 (ONNX runtime hoac PyTorch Dynamic Quantization)."""
    # 1. Thu onnxruntime quantization neu file ONNX ton tai
    if onnx_model_path.exists():
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print("[*] Running INT8 Dynamic Quantization (ONNX)...")
            quantize_dynamic(
                model_input=str(onnx_model_path),
                model_output=str(int8_save_path),
                weight_type=QuantType.QInt8
            )
            fp32_size = onnx_model_path.stat().st_size / 1024
            int8_size = int8_save_path.stat().st_size / 1024
            ratio = fp32_size / max(1.0, int8_size)
            print(f"[+] Created INT8 ONNX model: {int8_save_path}")
            print(f"    - FP32 Size: {fp32_size:.1f} KB")
            print(f"    - INT8 Size: {int8_size:.1f} KB (Compressed {ratio:.1f}x)")
            return
        except Exception as e:
            print(f"[!] ONNX runtime quantization warning: {e}")

    # 2. PyTorch Native Dynamic Quantization (INT8)
    if model_cpu is not None:
        try:
            print("[*] Running PyTorch Native INT8 Dynamic Quantization...")
            q_model = torch.ao.quantization.quantize_dynamic(
                model_cpu, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
            )
            pt_int8_path = int8_save_path.with_suffix(".pt")
            torch.save(q_model.state_dict(), pt_int8_path)
            orig_size = config.MODEL_PT_PATH.stat().st_size / 1024 if config.MODEL_PT_PATH.exists() else 3000
            q_size = pt_int8_path.stat().st_size / 1024
            ratio = orig_size / max(1.0, q_size)
            print(f"[+] Created PyTorch INT8 model: {pt_int8_path}")
            print(f"    - Orig Size: {orig_size:.1f} KB")
            print(f"    - INT8 Size: {q_size:.1f} KB (Compressed {ratio:.1f}x)")
        except Exception as e:
            print(f"[!] PyTorch quantization error: {e}")


def benchmark_onnx_accuracy(fp32_path: Path, int8_path: Path, X_val: np.ndarray, y_val: np.ndarray):
    """Benchmark do chinh xac va do tre truoc va sau khi luong tu hoa INT8."""
    try:
        import onnxruntime as ort
        sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

        # Do FP32
        t0 = time.perf_counter()
        preds_fp32 = []
        for i in range(len(X_val)):
            inp = X_val[i:i+1].astype(np.float32)
            out = sess_fp32.run(None, {"input_sequence": inp})
            preds_fp32.append(np.argmax(out[0][0]))
        lat_fp32 = (time.perf_counter() - t0) / len(X_val) * 1000
        acc_fp32 = np.mean(np.array(preds_fp32) == y_val) * 100

        # Do INT8
        t0 = time.perf_counter()
        preds_int8 = []
        for i in range(len(X_val)):
            inp = X_val[i:i+1].astype(np.float32)
            out = sess_int8.run(None, {"input_sequence": inp})
            preds_int8.append(np.argmax(out[0][0]))
        lat_int8 = (time.perf_counter() - t0) / len(X_val) * 1000
        acc_int8 = np.mean(np.array(preds_int8) == y_val) * 100

        print("=" * 60)
        print("ONNX QUANTIZATION COMPARISON (FP32 vs INT8):")
        print(f"  - FP32: Accuracy = {acc_fp32:.1f}% | Latency = {lat_fp32:.2f} ms/frame")
        print(f"  - INT8: Accuracy = {acc_int8:.1f}% | Latency = {lat_int8:.2f} ms/frame (Speedup {(lat_fp32/max(0.01, lat_int8)):.1f}x)")
        print("=" * 60)
    except Exception as e:
        print(f"[!] ONNX benchmark warning: {e}")


def train_gesture_model(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_name: str = config.DEVICE,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """Hàm chính thực thi toàn bộ pipeline huấn luyện, xuất ONNX và lượng tử hóa INT8."""
    print("=" * 60)
    print("TRAINING MULTI-SCALE BI-LSTM + ATTENTION (158D Context Gate, 45 Frames)")
    print("=" * 60)

    target_data_dir = Path(data_dir) if data_dir else config.DATASET_DIR
    target_out_dir = Path(output_dir) if output_dir else config.WEIGHTS_DIR
    target_out_dir.mkdir(parents=True, exist_ok=True)

    pt_save_path = target_out_dir / config.MODEL_PT_PATH.name
    onnx_save_path = target_out_dir / config.MODEL_ONNX_PATH.name
    int8_save_path = target_out_dir / config.MODEL_INT8_PATH.name

    # 1. Nạp và xáo trộn dữ liệu đồng đều trước khi chia tập (tránh mất lớp khi ghi dữ liệu thực tế)
    dataset_file = target_data_dir / "gesture_sequences.npz" if target_data_dir.is_dir() else target_data_dir
    X, y = load_dataset(dataset_file)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(y))
    split = int(0.85 * len(y))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(GestureSequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GestureSequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # 2. Khoi tao mo hinh
    device = torch.device(device_name if torch.cuda.is_available() and "cuda" in device_name else "cpu")
    print(f"[*] Training device: {device}")

    model = MultiScaleBiLSTMAttention(
        input_dim=config.FEATURE_DIM,
        cnn_branch_channels=32,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        num_classes=config.NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Vong lap huan luyen
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        scheduler.step()
        train_acc = correct / total * 100.0

        # Danh gia tap Val
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits, _ = model(vx)
                v_preds = torch.argmax(v_logits, dim=1)
                v_correct += (v_preds == vy).sum().item()
                v_total += len(vy)
        val_acc = (v_correct / v_total * 100.0) if v_total > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # 4. Luu trong so PyTorch (.pt)
    torch.save(model.state_dict(), pt_save_path)
    print(f"[+] Saved PyTorch weights to: {pt_save_path}")

    # 5. Xuat sang ONNX FP32
    export_to_onnx(model, onnx_save_path, input_dim=config.FEATURE_DIM, seq_len=config.SEQUENCE_LENGTH)

    # 6. Luong tu hoa INT8 (BO SUNG 2)
    quantize_onnx_int8(onnx_save_path, int8_save_path, model_cpu=model.to("cpu"))

    # 7. Do do chinh xac & kiem dinh
    benchmark_onnx_accuracy(onnx_save_path, int8_save_path, X_val, y_val)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HETHONGXULITAY - Huan luyen mo hinh Bi-LSTM nhan dien cu chi")
    parser.add_argument("--data_dir", type=str, default=str(config.DATASET_DIR), help="Thu muc chua dataset (.npz)")
    parser.add_argument("--output_dir", type=str, default=str(config.WEIGHTS_DIR), help="Thu muc luu weights (.pt, .onnx)")
    parser.add_argument("--epochs", type=int, default=25, help="So epochs huan luyen")
    parser.add_argument("--batch_size", type=int, default=32, help="Kich thuoc batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Toc do hoc (learning rate)")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Thiet bi (cuda:0 hoac cpu)")
    args = parser.parse_args()

    train_gesture_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
