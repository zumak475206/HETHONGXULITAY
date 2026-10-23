"""
src/ui/dialog_recorder.py
=========================
Hộp thoại thu thập mẫu cử chỉ trực quan chuyên nghiệp (Visual Recording Dialog).
Hiển thị webcam trực tiếp, khung xương bàn tay, đếm ngược 3-2-1 to rõ,
tùy chọn thời lượng (1.5s, 3.0s, 5.0s, tự do), tự động nắn chuẩn (Temporal Resampling)
về 45 frames x 158D và hỗ trợ xóa ngay mẫu vừa quay nếu bị lỗi.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import config
from src.database.dataset_merger import delete_sample_file, resample_sequence, save_single_sample
from src.database.db_manager import DatabaseManager
from src.ui.theme import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_ROSE,
    DARK_BG_CARD,
    DARK_BG_SURFACE,
    DARK_BORDER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class RecordingDialog(QDialog):
    """
    Hộp thoại thu thập mẫu chuỗi cử chỉ thời gian thực với khung hình Camera trực tiếp.
    """

    sample_recorded = pyqtSignal(str, str)  # (gesture_name, file_path)
    sample_deleted = pyqtSignal(str, str)   # (gesture_name, file_path)

    def __init__(
        self,
        db: DatabaseManager,
        gesture_info: Dict[str, Any],
        author: str = "Tôi",
        camera_worker: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.gesture_info = gesture_info
        self.gesture_name = str(gesture_info.get("name", "UNKNOWN")).strip().upper()
        self.display_name = str(gesture_info.get("display_name", self.gesture_name))
        self.finger_guide = str(gesture_info.get("finger_guide", "Thực hiện tư thế tay theo chuẩn"))
        self.mode_id = int(gesture_info.get("mode_id", 2))
        self.active_author = author
        self.camera_worker = camera_worker

        # Trạng thái ghi mẫu
        self._state = "IDLE"  # "IDLE" | "COUNTDOWN" | "RECORDING"
        self._countdown_val = 3
        self._record_start_time = 0.0
        self._target_duration_sec = 3.0  # 1.5, 3.0, 5.0, hoặc 0.0 (tự do)
        self._record_buffer: List[List[float]] = []
        self._last_saved_file: Optional[Path] = None
        self._session_count = 0

        # Timers
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(750)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._record_timer = QTimer(self)
        self._record_timer.setInterval(33)  # ~30 FPS UI tick
        self._record_timer.timeout.connect(self._on_record_tick)

        self._setup_ui()
        self._connect_camera()
        self._update_counter_label()

        # Tạm dừng điều khiển chuột OS và đặt recording mode_id cho camera_worker
        if self.camera_worker:
            if hasattr(self.camera_worker, "set_recording_override"):
                self.camera_worker.set_recording_override(True, mode_id=self.mode_id)
            elif hasattr(self.camera_worker, "set_mouse_paused"):
                self.camera_worker.set_mouse_paused(True)

    def _setup_ui(self) -> None:
        """Khởi tạo giao diện hộp thoại hiện đại, sắc nét."""
        self.setWindowTitle(f"Phòng Thu Thập Mẫu — {self.gesture_name}")
        self.setMinimumSize(820, 680)
        self.resize(840, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_BG_SURFACE};
                color: {TEXT_PRIMARY};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # ── 1. HEADER: TÊN CỬ CHỈ & HƯỚNG DẪN ────────────────────────────────
        hdr_frame = QFrame()
        hdr_frame.setStyleSheet(f"""
            background-color: {DARK_BG_CARD};
            border: 1px solid {DARK_BORDER};
            border-radius: 10px;
            padding: 10px 16px;
        """)
        hdr_layout = QVBoxLayout(hdr_frame)
        hdr_layout.setContentsMargins(12, 10, 12, 10)
        hdr_layout.setSpacing(4)

        top_row = QHBoxLayout()
        lbl_gname = QLabel(f"🎯 {self.gesture_name} — {self.display_name}")
        lbl_gname.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 15px; font-weight: 800; letter-spacing: 0.5px;")
        top_row.addWidget(lbl_gname)
        top_row.addStretch()

        mode_names = {
            1: "MODE 1: CHUỘT",
            2: "MODE 2: TRÌNH CHIẾU",
            3: "MODE 3: KÝ HIỆU NNKH",
        }
        mode_text = mode_names.get(self.mode_id, f"MODE {self.mode_id}")
        lbl_mode = QLabel(mode_text)
        lbl_mode.setStyleSheet(f"""
            background-color: rgba(16, 185, 129, 0.20);
            color: {COLOR_GREEN};
            border: 1px solid {COLOR_GREEN};
            border-radius: 10px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 700;
        """)
        top_row.addWidget(lbl_mode)
        hdr_layout.addLayout(top_row)

        lbl_guide = QLabel(f"☝️ Hướng dẫn: {self.finger_guide}")
        lbl_guide.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 500;")
        hdr_layout.addWidget(lbl_guide)

        layout.addWidget(hdr_frame)

        # ── 2. KHUNG CAMERA PREVIEW CANVAS ────────────────────────────────────
        self.cam_container = QFrame()
        self.cam_container.setProperty("class", "CardFrame")
        self.cam_container.setStyleSheet(f"""
            background-color: #0b1120;
            border: 2px solid {DARK_BORDER};
            border-radius: 12px;
        """)
        cam_layout = QVBoxLayout(self.cam_container)
        cam_layout.setContentsMargins(4, 4, 4, 4)
        cam_layout.setSpacing(0)

        self.lbl_camera = QLabel()
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setMinimumSize(640, 380)
        self.lbl_camera.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_camera.setText("📷 Đang đồng bộ luồng video webcam...")
        self.lbl_camera.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px; font-weight: 600;")
        cam_layout.addWidget(self.lbl_camera)

        layout.addWidget(self.cam_container, stretch=1)

        # ── 3. THANH ĐIỀU KHIỂN & TÙY CHỌN THỜI LƯỢNG ─────────────────────────
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(f"""
            background-color: {DARK_BG_CARD};
            border: 1px solid {DARK_BORDER};
            border-radius: 10px;
            padding: 8px 14px;
        """)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(10)

        # Hàng chọn thời lượng & số lượng mẫu
        opt_row = QHBoxLayout()
        lbl_dur = QLabel("Thời lượng quay:")
        lbl_dur.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;")
        opt_row.addWidget(lbl_dur)

        self.cbo_duration = QComboBox()
        self.cbo_duration.addItems([
            "Chuẩn: 3.0 giây (Khuyến nghị cho NNKH)",
            "Nhanh: 1.5 giây (Cử chỉ gạt/lướt ngắn)",
            "Dài: 5.0 giây (Ký hiệu phức tạp/chậm)",
            "Tự do: Bấm Dừng Thủ Công",
        ])
        self.cbo_duration.setMinimumWidth(260)
        self.cbo_duration.currentIndexChanged.connect(self._on_duration_changed)
        opt_row.addWidget(self.cbo_duration)

        opt_row.addStretch()

        self.lbl_counter = QLabel("📦 Phiên này: 0 mẫu | Tổng trong DB: 0")
        self.lbl_counter.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px; font-weight: 700;")
        opt_row.addWidget(self.lbl_counter)
        ctrl_layout.addLayout(opt_row)

        # Thanh tiến trình thời gian thực
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        # Nhãn trạng thái
        self.lbl_status = QLabel("Sẵn sàng. Nhìn vào camera và bấm 'Bắt Đầu Quay Mẫu'.")
        self.lbl_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-style: italic;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl_layout.addWidget(self.lbl_status)

        # Hàng nút bấm thao tác
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_record = QPushButton("🔴 Bắt Đầu Quay Mẫu (Đếm 3-2-1)")
        self.btn_record.setProperty("class", "PrimaryButton")
        self.btn_record.setMinimumHeight(42)
        self.btn_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_record.clicked.connect(self.start_recording)
        btn_row.addWidget(self.btn_record, stretch=3)

        self.btn_stop = QPushButton("⏹ Dừng & Lưu Ngay")
        self.btn_stop.setProperty("class", "SecondaryButton")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_and_save)
        btn_row.addWidget(self.btn_stop, stretch=2)

        self.btn_delete_last = QPushButton("🗑️ Xóa Mẫu Vừa Quay")
        self.btn_delete_last.setMinimumHeight(42)
        self.btn_delete_last.setEnabled(False)
        self.btn_delete_last.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_last.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(244, 63, 94, 0.15);
                color: {COLOR_ROSE};
                border: 1px solid rgba(244, 63, 94, 0.40);
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: rgba(244, 63, 94, 0.30);
                border-color: {COLOR_ROSE};
            }}
            QPushButton:disabled {{
                background-color: rgba(255, 255, 255, 0.04);
                color: {TEXT_MUTED};
                border-color: rgba(255, 255, 255, 0.08);
            }}
        """)
        self.btn_delete_last.clicked.connect(self.delete_last_sample)
        btn_row.addWidget(self.btn_delete_last, stretch=2)

        self.btn_close = QPushButton("✖️ Đóng")
        self.btn_close.setProperty("class", "SecondaryButton")
        self.btn_close.setMinimumHeight(42)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close, stretch=1)

        ctrl_layout.addLayout(btn_row)

        lbl_hint = QLabel("💡 Mẹo: Nhấn phím [SPACE / CÁCH] để Bắt Đầu hoặc Dừng && Lưu ngay | [ESC] Đóng")
        lbl_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl_layout.addWidget(lbl_hint)

        layout.addWidget(ctrl_frame)

    def _connect_camera(self) -> None:
        """Kết nối luồng camera để nhận frame và feature vectors."""
        if not self.camera_worker:
            return

        try:
            self.camera_worker.frame_ready.connect(self.on_camera_frame)
            self.camera_worker.feature_ready.connect(self.on_feature_frame)
        except Exception as e:
            print(f"[RecordingDialog] Lỗi kết nối camera: {e}")

    def _disconnect_camera(self) -> None:
        """Ngắt kết nối an toàn khi đóng hộp thoại."""
        if not self.camera_worker:
            return

        try:
            self.camera_worker.frame_ready.disconnect(self.on_camera_frame)
        except Exception:
            pass
        try:
            self.camera_worker.feature_ready.disconnect(self.on_feature_frame)
        except Exception:
            pass

    def _on_duration_changed(self, idx: int) -> None:
        """Thay đổi thời lượng thu thập mẫu theo preset."""
        presets = [3.0, 1.5, 5.0, 0.0]
        if 0 <= idx < len(presets):
            self._target_duration_sec = presets[idx]

    def _update_counter_label(self) -> None:
        """Cập nhật bộ đếm mẫu đã quay phiên này và tổng trong database."""
        try:
            counts = self.db.get_gesture_sample_counts()
            total_db = counts.get(self.gesture_name, 0)
        except Exception:
            total_db = 0

        self.lbl_counter.setText(f"📦 Phiên này: {self._session_count} mẫu | Tổng trong DB: {total_db}")

    # ── XỬ LÝ KHUNG HÌNH CAMERA & HIỆU ỨNG TRỰC QUAN ─────────────────────────

    def on_camera_frame(self, frame: np.ndarray) -> None:
        """Hiển thị khung hình camera kèm hiệu ứng đếm ngược và trạng thái REC."""
        if frame is None or not self.isVisible():
            return

        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        if self._state == "COUNTDOWN":
            # Vẽ vòng tròn mờ và số đếm to ở giữa màn hình
            overlay = display_frame.copy()
            cv2.circle(overlay, (w // 2, h // 2), 70, (15, 23, 42), -1)
            cv2.circle(overlay, (w // 2, h // 2), 70, (0, 215, 255), 4)
            cv2.addWeighted(overlay, 0.70, display_frame, 0.30, 0, display_frame)

            txt = str(self._countdown_val)
            t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.0, 5)[0]
            cv2.putText(
                display_frame,
                txt,
                (w // 2 - t_size[0] // 2, h // 2 + t_size[1] // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                3.0,
                (0, 230, 255),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "CHUAN BI GIO TAY...",
                (w // 2 - 110, h // 2 + 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        elif self._state == "RECORDING":
            # Viền đỏ phát sáng bao quanh khung hình
            border_th = 5
            cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 0, 240), border_th)

            # Badge REC nhấp nháy góc trên bên trái
            elapsed = time.time() - self._record_start_time
            cv2.rectangle(display_frame, (16, 16), (220, 52), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (16, 16), (220, 52), (0, 0, 240), 2)
            cv2.circle(display_frame, (32, 34), 7, (0, 0, 255), -1)

            dur_str = f"{self._target_duration_sec:.1f}s" if self._target_duration_sec > 0 else "TU DO"
            rec_text = f"REC {elapsed:.1f}s / {dur_str}"
            cv2.putText(display_frame, rec_text, (48, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Số frame thu được góc trên bên phải
            n_frames = len(self._record_buffer)
            cv2.rectangle(display_frame, (w - 140, 16), (w - 16, 52), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (w - 140, 16), (w - 16, 52), (0, 220, 255), 1)
            cv2.putText(display_frame, f"{n_frames} FRAMES", (w - 128, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 2)

        # Chuyển đổi sang QPixmap và render lên QLabel
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)

        # Scale vừa vặn kích thước hiển thị
        lbl_size = self.lbl_camera.size()
        scaled = pix.scaled(
            lbl_size.width(),
            lbl_size.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_camera.setPixmap(scaled)

    def on_feature_frame(self, feat_158d: list) -> None:
        """Thu thập vector 158D khi đang ở trạng thái RECORDING."""
        if self._state == "RECORDING":
            self._record_buffer.append(list(feat_158d))

    def keyPressEvent(self, event) -> None:
        """Hỗ trợ phím tắt SPACE để bắt đầu / dừng lưu ngay và ESC để hủy / đóng."""
        if event.key() == Qt.Key.Key_Space:
            if self._state == "IDLE":
                self.start_recording()
            elif self._state in ["COUNTDOWN", "RECORDING"]:
                self.stop_and_save()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Escape:
            if self._state in ["COUNTDOWN", "RECORDING"]:
                self.stop_and_save()
            else:
                self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    # ── LOGIC CHU TRÌNH THU MẪU ──────────────────────────────────────────────

    def start_recording(self) -> None:
        """Bắt đầu chu trình đếm ngược 3... 2... 1..."""
        self._record_buffer = []
        self._countdown_val = 3
        self._state = "COUNTDOWN"

        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_stop.setText("❌ Hủy Đếm Ngược (Space)")
        self.btn_delete_last.setEnabled(False)
        self.cbo_duration.setEnabled(False)
        self.cam_container.setStyleSheet(f"border: 2px solid {COLOR_AMBER}; border-radius: 12px;")

        self.lbl_status.setText(f"⏳ Chuẩn bị: Đếm ngược {self._countdown_val}... (Nhấn Space để hủy)")
        self.lbl_status.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 13px; font-weight: 700;")
        self.progress_bar.setValue(0)

        self._countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        """Xử lý từng nhịp đếm ngược."""
        self._countdown_val -= 1
        if self._countdown_val > 0:
            self.lbl_status.setText(f"⏳ Chuẩn bị: Đếm ngược {self._countdown_val}... (Nhấn Space để hủy)")
        else:
            self._countdown_timer.stop()
            self._state = "RECORDING"
            self._record_start_time = time.time()
            self.btn_stop.setText("⏹ Dừng && Lưu Ngay (Space)")
            self.btn_stop.setEnabled(True)
            self.cam_container.setStyleSheet(f"border: 3px solid {COLOR_ROSE}; border-radius: 12px;")
            self.lbl_status.setText("🔴 ĐANG THU THẬP MẪU... Hãy thực hiện cử chỉ trước camera! (Bấm Space để dừng & lưu)")
            self.lbl_status.setStyleSheet(f"color: {COLOR_ROSE}; font-size: 13px; font-weight: 700;")
            self._record_timer.start()

    def _on_record_tick(self) -> None:
        """Cập nhật thanh tiến trình theo thời gian và tự ngắt khi đủ thời lượng."""
        if self._state != "RECORDING":
            return

        elapsed = time.time() - self._record_start_time
        if self._target_duration_sec > 0:
            pct = int((elapsed / max(0.1, self._target_duration_sec)) * 100)
            self.progress_bar.setValue(min(100, pct))
            self.lbl_status.setText(f"🔴 Đang quay cử chỉ: {elapsed:.1f}s / {self._target_duration_sec:.1f}s ({len(self._record_buffer)} frames)")

            if elapsed >= self._target_duration_sec:
                self.stop_and_save()
        else:
            self.progress_bar.setValue(min(100, int((elapsed % 2.0) * 50)))
            self.lbl_status.setText(f"🔴 Đang quay (Tự do): {elapsed:.1f}s ({len(self._record_buffer)} frames) — Bấm 'Dừng && Lưu Ngay' hoặc Space khi làm xong")

    def stop_and_save(self) -> None:
        """Dừng quay, nắn chuẩn thời gian về 45 frames và lưu file .npz vào SQLite."""
        if self._state == "COUNTDOWN":
            self._countdown_timer.stop()
            self._state = "IDLE"
            self.btn_record.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("⏹ Dừng && Lưu Ngay")
            self.cbo_duration.setEnabled(True)
            self.cam_container.setStyleSheet(f"border: 2px solid {DARK_BORDER}; border-radius: 12px;")
            self.lbl_status.setText("Đã hủy đếm ngược. Nhìn vào camera và bấm 'Bắt Đầu Quay Mẫu' hoặc nhấn Space.")
            self.lbl_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-style: italic;")
            self.progress_bar.setValue(0)
            return

        if self._state != "RECORDING":
            return

        self._record_timer.stop()
        self._countdown_timer.stop()
        self._state = "IDLE"

        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("⏹ Dừng && Lưu Ngay")
        self.cbo_duration.setEnabled(True)
        self.cam_container.setStyleSheet(f"border: 2px solid {DARK_BORDER}; border-radius: 12px;")

        # Kiểm tra số lượng frame tối thiểu
        if len(self._record_buffer) < 5:
            self.lbl_status.setText("⚠️ Chưa đủ khung hình (tối thiểu 5 frames). Vui lòng giơ tay và quay lại!")
            self.lbl_status.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 12px; font-weight: 700;")
            self.progress_bar.setValue(0)
            return

        # Nắn chuẩn thời gian (Temporal Resampling) về 45 frames x 158D
        raw_arr = np.array(self._record_buffer, dtype=np.float32)
        resampled_arr = resample_sequence(raw_arr, target_seq_len=45)

        try:
            saved_path = save_single_sample(
                sequence=resampled_arr,
                gesture_name=self.gesture_name,
                author=self.active_author,
                db=self.db,
                auto_resample=True,
                target_seq_len=45,
            )
            self._last_saved_file = Path(saved_path)
            self._session_count += 1
            self.btn_delete_last.setEnabled(True)
            self.progress_bar.setValue(100)

            self.lbl_status.setText(
                f"✅ Đã lưu mẫu {self._session_count}! ({len(self._record_buffer)} frames thô → 45 frames chuẩn). Sẵn sàng quay tiếp."
            )
            self.lbl_status.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px; font-weight: 700;")
            self._update_counter_label()

            # Phát tín hiệu ra ngoài để Studio & Library cập nhật
            self.sample_recorded.emit(self.gesture_name, str(self._last_saved_file))

        except Exception as e:
            self.lbl_status.setText(f"❌ Lỗi lưu mẫu: {e}")
            self.lbl_status.setStyleSheet(f"color: {COLOR_ROSE}; font-size: 12px; font-weight: 700;")

    def delete_last_sample(self) -> None:
        """Xóa ngay lập tức mẫu vừa thu thập gần nhất nếu quay bị lỗi hoặc sai động tác."""
        if not self._last_saved_file or not self._last_saved_file.exists():
            return

        file_to_del = str(self._last_saved_file)
        success = delete_sample_file(file_to_del, db=self.db)

        if success:
            self._session_count = max(0, self._session_count - 1)
            self.btn_delete_last.setEnabled(False)
            self._last_saved_file = None
            self._update_counter_label()

            self.lbl_status.setText("🗑️ Đã xóa mẫu vừa quay thành công! Bạn có thể bấm quay lại ngay.")
            self.lbl_status.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 12px; font-weight: 700;")
            self.progress_bar.setValue(0)

            self.sample_deleted.emit(self.gesture_name, file_to_del)
        else:
            QMessageBox.warning(self, "Lỗi xóa", f"Không thể xóa file: {file_to_del}")

    def closeEvent(self, event) -> None:
        """Dọn dẹp timer và ngắt kết nối khi đóng hộp thoại."""
        self._countdown_timer.stop()
        self._record_timer.stop()
        self._disconnect_camera()
        if self.camera_worker:
            if hasattr(self.camera_worker, "set_recording_override"):
                self.camera_worker.set_recording_override(False)
            elif hasattr(self.camera_worker, "set_mouse_paused"):
                self.camera_worker.set_mouse_paused(False)
        event.accept()
        self._countdown_timer.stop()
        self._record_timer.stop()
        self._disconnect_camera()
        event.accept()
