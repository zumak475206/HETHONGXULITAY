"""
src/database/dataset_merger.py
==============================
Bộ nạp và đồng bộ dữ liệu cử chỉ phân tán (Modular Dataset Merger & Synchronizer).
Cho phép nhiều thành viên cùng thu thập mẫu độc lập mà không bị xung đột file,
hỗ trợ đóng gói xuất/nhập ZIP và gộp thành ma trận tensor huấn luyện PyTorch.
"""

import os
import zipfile
import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

import config
from src.database.db_manager import DatabaseManager


SEQUENCES_DIR = config.DATASET_DIR / "sequences"
LEGACY_DATASET_FILE = config.DATASET_DIR / "gesture_sequences.npz"


def resample_sequence(sequence: np.ndarray, target_seq_len: int = 45) -> np.ndarray:
    """
    Nắn chuẩn thời gian (Temporal Resampling / Linear Interpolation) một chuỗi cử chỉ
    có độ dài bất kỳ (N frames) về đúng chuẩn target_seq_len (mặc định 45 frames).
    Bảo toàn 100% động thái chuyển động ngón tay, bất biến với tốc độ thao tác.
    """
    arr = np.array(sequence, dtype=np.float32)
    if arr.ndim != 2:
        return arr

    curr_len, curr_dim = arr.shape
    if curr_len == target_seq_len:
        return arr
    if curr_len < 2:
        return np.repeat(arr, target_seq_len, axis=0)[:target_seq_len]

    x_old = np.linspace(0.0, 1.0, curr_len)
    x_new = np.linspace(0.0, 1.0, target_seq_len)
    resampled = np.zeros((target_seq_len, curr_dim), dtype=np.float32)
    for d in range(curr_dim):
        resampled[:, d] = np.interp(x_new, x_old, arr[:, d])
    return resampled


def delete_sample_file(
    file_path: Any,
    db: Optional[DatabaseManager] = None,
) -> bool:
    """
    Xóa một file mẫu .npz trên ổ đĩa và gỡ bản ghi tương ứng khỏi SQLite database.
    """
    p = Path(file_path)
    file_existed = False
    if p.exists():
        try:
            p.unlink()
            file_existed = True
        except Exception:
            pass

    db_deleted = False
    if db is not None:
        db_deleted = db.delete_sample(str(p))

    return file_existed or db_deleted


def save_single_sample(
    sequence: np.ndarray,
    gesture_name: str,
    author: str = "User",
    output_dir: Optional[Path] = None,
    db: Optional[DatabaseManager] = None,
    auto_resample: bool = True,
    target_seq_len: int = 45,
) -> Path:
    """
    Lưu một chuỗi cử chỉ đơn lẻ (shape T, D) thành file .npz theo chuẩn:
    dataset/sequences/<gesture_name>/<author>_<timestamp>.npz
    Tự động nội suy nắn chuẩn về target_seq_len nếu auto_resample=True.
    """
    target_dir = Path(output_dir) if output_dir else SEQUENCES_DIR
    gesture_dir = target_dir / gesture_name.strip().upper()
    gesture_dir.mkdir(parents=True, exist_ok=True)

    # Đảm bảo tensor có đúng số chiều và được nắn chuẩn
    arr = np.array(sequence, dtype=np.float32)
    if auto_resample and arr.ndim == 2 and arr.shape[0] != target_seq_len:
        arr = resample_sequence(arr, target_seq_len=target_seq_len)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{author}_{timestamp_str}.npz"
    file_path = gesture_dir / filename

    np.savez_compressed(
        file_path,
        sequence=arr,
        gesture_name=gesture_name.strip().upper(),
        author=author,
        timestamp=timestamp_str,
    )

    # Đăng ký vào database nếu có db manager
    if db is not None:
        db.add_sample(
            gesture_name=gesture_name.strip().upper(),
            file_path=str(file_path),
            author=author,
            num_frames=int(arr.shape[0]) if len(arr.shape) >= 1 else target_seq_len,
            feature_dim=int(arr.shape[-1]) if len(arr.shape) >= 2 else 158,
        )

    return file_path


def sync_folder_with_db(
    db: DatabaseManager,
    sequences_dir: Optional[Path] = None,
) -> int:
    """
    Quét toàn bộ thư mục sequences/ và tự động cập nhật các file mẫu mới vào database SQLite.
    Trả về số lượng mẫu mới được đồng bộ.
    """
    seq_dir = Path(sequences_dir) if sequences_dir else SEQUENCES_DIR
    if not seq_dir.exists():
        return 0

    existing_samples = {s["file_path"] for s in db.get_all_samples()}
    synced_count = 0

    for npz_file in seq_dir.rglob("*.npz"):
        file_str = str(npz_file)
        if file_str not in existing_samples:
            try:
                data = np.load(npz_file, allow_pickle=True)
                g_name = str(data.get("gesture_name", npz_file.parent.name)).strip().upper()
                author = str(data.get("author", "User"))
                seq = data.get("sequence", None)
                n_frames = int(seq.shape[0]) if seq is not None and len(seq.shape) >= 1 else 45
                feat_dim = int(seq.shape[-1]) if seq is not None and len(seq.shape) >= 2 else 158

                # Đảm bảo cử chỉ có trong bảng gestures
                if not db.get_gesture(g_name):
                    db.add_gesture(
                        name=g_name,
                        display_name=g_name,
                        mode_id=1,
                        description="Cử chỉ nạp từ thư mục dữ liệu",
                    )

                db.add_sample(
                    gesture_name=g_name,
                    file_path=file_str,
                    author=author,
                    num_frames=n_frames,
                    feature_dim=feat_dim,
                )
                synced_count += 1
            except Exception:
                continue

    return synced_count


def load_merged_dataset(
    db: Optional[DatabaseManager] = None,
    sequences_dir: Optional[Path] = None,
    legacy_file: Optional[Path] = None,
    include_legacy: bool = True,
    target_seq_len: int = 45,
    target_dim: int = 158,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Tải và gộp toàn bộ dữ liệu từ:
    1. Các file mẫu phân tán (.npz) trong dataset/sequences/
    2. File dataset kế thừa (gesture_sequences.npz nếu có)

    Trả về:
    - X: np.ndarray shape (N, target_seq_len, target_dim)
    - y: np.ndarray shape (N,)
    - class_names: Danh sách tên lớp theo đúng thứ tự nhãn
    """
    all_sequences = []
    all_labels = []

    # 1. Thu thập từ các file mẫu phân tán
    seq_dir = Path(sequences_dir) if sequences_dir else SEQUENCES_DIR
    if seq_dir.exists():
        for npz_file in seq_dir.rglob("*.npz"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                seq = data.get("sequence", None)
                if seq is None or len(seq.shape) != 2:
                    continue
                g_name = str(data.get("gesture_name", npz_file.parent.name)).strip().upper()

                # Resample số frame về target_seq_len nếu cần
                curr_len, curr_dim = seq.shape
                if curr_len != target_seq_len:
                    x_old = np.linspace(0, 1, curr_len)
                    x_new = np.linspace(0, 1, target_seq_len)
                    resampled = np.zeros((target_seq_len, curr_dim), dtype=np.float32)
                    for d in range(curr_dim):
                        resampled[:, d] = np.interp(x_new, x_old, seq[:, d])
                    seq = resampled

                # Điều chỉnh số chiều về target_dim
                if curr_dim < target_dim:
                    pad = np.zeros((target_seq_len, target_dim - curr_dim), dtype=np.float32)
                    seq = np.concatenate([seq, pad], axis=-1)
                elif curr_dim > target_dim:
                    seq = seq[:, :target_dim]

                all_sequences.append(seq)
                all_labels.append(g_name)
            except Exception:
                continue

    # 2. Thu thập từ file legacy (nếu include_legacy=True)
    if include_legacy:
        leg_path = Path(legacy_file) if legacy_file else LEGACY_DATASET_FILE
        if leg_path.exists():
            try:
                data = np.load(leg_path, allow_pickle=True)
                if "X" in data and "y" in data:
                    leg_X = data["X"]
                    leg_y = data["y"]
                    # Chuẩn hóa shape
                    for i in range(len(leg_y)):
                        sample = leg_X[i]
                        curr_len, curr_dim = sample.shape
                        if curr_len != target_seq_len:
                            x_old = np.linspace(0, 1, curr_len)
                            x_new = np.linspace(0, 1, target_seq_len)
                            resampled = np.zeros((target_seq_len, curr_dim), dtype=np.float32)
                            for d in range(curr_dim):
                                resampled[:, d] = np.interp(x_new, x_old, sample[:, d])
                            sample = resampled
                        if curr_dim < target_dim:
                            pad = np.zeros((target_seq_len, target_dim - curr_dim), dtype=np.float32)
                            sample = np.concatenate([sample, pad], axis=-1)
                        elif curr_dim > target_dim:
                            sample = sample[:, :target_dim]

                        cls_name = config.GESTURE_CLASSES[int(leg_y[i])] if int(leg_y[i]) < len(config.GESTURE_CLASSES) else f"GESTURE_{leg_y[i]}"
                        all_sequences.append(sample)
                        all_labels.append(cls_name)
            except Exception:
                pass

    if not all_sequences:
        # Nếu chưa có mẫu nào, sinh dataset giả lập để không sập ứng dụng
        return np.zeros((0, target_seq_len, target_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64), config.GESTURE_CLASSES

    # Xây dựng danh sách lớp duy nhất
    class_set = sorted(list(set(all_labels)))
    # Đảm bảo các lớp cốt lõi luôn ở đầu danh sách theo đúng thứ tự config
    ordered_classes = []
    for g in config.GESTURE_CLASSES:
        if g in class_set:
            ordered_classes.append(g)
    for g in class_set:
        if g not in ordered_classes:
            ordered_classes.append(g)

    label_to_idx = {name: idx for idx, name in enumerate(ordered_classes)}
    y_indices = np.array([label_to_idx[name] for name in all_labels], dtype=np.int64)
    X_tensor = np.array(all_sequences, dtype=np.float32)

    return X_tensor, y_indices, ordered_classes


def export_contributor_zip(
    author: str,
    output_zip_path: Path,
    sequences_dir: Optional[Path] = None,
) -> int:
    """
    Đóng gói toàn bộ file mẫu do tác giả `author` thu thập thành file .zip để chia sẻ.
    Trả về số lượng mẫu đã nén.
    """
    seq_dir = Path(sequences_dir) if sequences_dir else SEQUENCES_DIR
    out_path = Path(output_zip_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if seq_dir.exists():
            for npz_file in seq_dir.rglob("*.npz"):
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    if str(data.get("author", "")).strip().lower() == author.strip().lower():
                        arcname = npz_file.relative_to(seq_dir)
                        zf.write(npz_file, arcname=str(arcname))
                        count += 1
                except Exception:
                    continue
    return count


def import_contributor_zip(
    zip_path: Path,
    target_sequences_dir: Optional[Path] = None,
    db: Optional[DatabaseManager] = None,
) -> int:
    """
    Giải nén file .zip chứa các mẫu thu thập từ bạn bè vào thư mục dữ liệu,
    tự động đồng bộ vào database. Trả về số lượng mẫu mới được nhập.
    """
    z_path = Path(zip_path)
    if not z_path.exists():
        return 0

    dest_dir = Path(target_sequences_dir) if target_sequences_dir else SEQUENCES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(z_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".npz"):
                zf.extract(member, dest_dir)
                count += 1

    if db is not None:
        sync_folder_with_db(db, dest_dir)

    return count
