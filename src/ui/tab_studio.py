"""
src/ui/tab_studio.py
====================
Tab 3 — Studio Thu Thập, Thêm Ký Hiệu Mới & Huấn Luyện AI cho HETHONGXULITAY.

Chứa:
  - StudioPage(QWidget):
      1. Form thêm ký hiệu / cử chỉ mới (Tên mã, tên hiển thị, mode, hướng dẫn ngón tay).
      2. Bộ thu thập mẫu trực tiếp trên App (Đếm ngược 3-2-1, gom 45 frames 158D, lưu .npz).
      3. Quản lý gói dữ liệu phân tán (Xuất / Nhập file ZIP chia sẻ bạn bè).
      4. Bảng điều khiển huấn luyện AI Hot-Reload (Chạy TrainWorker QThread, thanh tiến trình, xuất weights).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import config
from src.database.dataset_merger import (
    export_contributor_zip,
    import_contributor_zip,
    save_single_sample,
)
from src.ui.dialog_recorder import RecordingDialog
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
from src.ui.workers import TrainWorker


_TARGET_SEQ_LEN = config.SEQUENCE_LENGTH  # 45 frames


class StudioPage(QWidget):
    """
    Trang Studio hoàn chỉnh (Tab 3).
    Quản trị thu thập mẫu, thêm ký hiệu mới và Huấn luyện AI Hot-Reload.
    """

    gesture_added = pyqtSignal(str)       # Phát ra khi thêm mới cử chỉ thành công
    sample_recorded = pyqtSignal(str)     # Phát ra khi lưu thành công 1 mẫu (45 frames)
    training_finished = pyqtSignal(dict)  # Phát ra khi huấn luyện AI hoàn tất

    def __init__(
        self,
        db,
        author: str = "Tôi",
        camera_worker=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.active_author = author
        self.camera_worker = camera_worker
        self.train_worker: Optional[TrainWorker] = None

        # Trạng thái ghi mẫu cử chỉ
        self._is_recording = False
        self._countdown_val = 0
        self._record_buffer: List[List[float]] = []

        # Timer đếm ngược 3... 2... 1...
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(700)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()
        self.refresh_gestures_dropdown()

        # Kết nối tới CameraWorker nếu đã có
        if self.camera_worker is not None:
            self.connect_camera_worker(self.camera_worker)

    def connect_camera_worker(self, worker) -> None:
        """Kết nối luồng camera để nhận feature vector 158D thời gian thực."""
        self.camera_worker = worker
        if hasattr(worker, "feature_ready"):
            try:
                worker.feature_ready.connect(self.on_feature_frame)
            except Exception:
                pass

    def set_active_author(self, author: str) -> None:
        """Cập nhật tên người thu thập mẫu."""
        self.active_author = author
        if hasattr(self, "lbl_author_info"):
            self.lbl_author_info.setText(f"👤 Tác giả thu thập: {author}")

    # ── XÂY DỰNG GIAO DIỆN ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # ── Tiêu đề & Nút xuất/nhập ZIP nhanh ────────────────────────────────
        header_row = QHBoxLayout()
        lbl_title = QLabel("PHÒNG THU THẬP & HUẤN LUYỆN MÔ HÌNH (STUDIO)")
        lbl_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 700;")
        header_row.addWidget(lbl_title)
        header_row.addStretch()

        btn_top_export = QPushButton("📦 Xuất Gói Của Tôi (.ZIP)")
        btn_top_export.setProperty("class", "SecondaryButton")
        btn_top_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_top_export.clicked.connect(self.export_my_zip)

        btn_top_import = QPushButton("📥 Nhập Gói Bạn Bè (.ZIP)")
        btn_top_import.setProperty("class", "SecondaryButton")
        btn_top_import.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_top_import.clicked.connect(self.import_friend_zip)

        header_row.addWidget(btn_top_export)
        header_row.addWidget(btn_top_import)
        root.addLayout(header_row)

        # ── Scroll Area bọc 2 cột nội dung chính ──────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(16)

        # ── CỘT TRÁI: Đăng Ký Cử Chỉ Mới + Bộ Thu Thập Mẫu Trực Tiếp ─────────
        left_col = QVBoxLayout()
        left_col.setSpacing(14)

        card_add = self._build_card_add_gesture()
        card_record = self._build_card_recorder()

        left_col.addWidget(card_add)
        left_col.addWidget(card_record)
        container_layout.addLayout(left_col, stretch=1)

        # ── CỘT PHẢI: Huấn Luyện AI Hot-Reload + Quản Lý Gói ZIP ───────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(14)

        card_train = self._build_card_training()
        card_packages = self._build_card_packages()

        right_col.addWidget(card_train)
        right_col.addWidget(card_packages)
        container_layout.addLayout(right_col, stretch=1)

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

    # ── 1. CARD: ĐĂNG KÝ CỬ CHỈ MỚI ──────────────────────────────────────────

    def _build_card_add_gesture(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        lbl_t = QLabel("1. ĐĂNG KÝ KÝ HIỆU / CỬ CHỈ MỚI")
        lbl_t.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;")
        layout.addWidget(lbl_t)

        lbl_sub = QLabel("Thêm cử chỉ mới vào cơ sở dữ liệu SQLite để bắt đầu quay mẫu và huấn luyện.")
        lbl_sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(lbl_sub)

        # Form fields
        self.txt_gesture_name = QLineEdit()
        self.txt_gesture_name.setPlaceholderText("Tên mã (VD: VSL_HELLO, ASL_A, SWIPE_CIRCLE)")

        self.txt_display_name = QLineEdit()
        self.txt_display_name.setPlaceholderText("Tên hiển thị tiếng Việt (VD: Xin chào - Ngôn ngữ ký hiệu)")

        mode_row = QHBoxLayout()
        lbl_mode_caption = QLabel("Chế độ áp dụng:")
        lbl_mode_caption.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        self.cbo_mode = QComboBox()
        self.cbo_mode.addItems([
            "Mode 1: Chuột (Mouse)",
            "Mode 2: Trình chiếu / Media",
            "Mode 3: Ngôn ngữ ký hiệu (VSL)",
        ])
        self.cbo_mode.setCurrentIndex(2)  # Mặc định Mode 3
        mode_row.addWidget(lbl_mode_caption)
        mode_row.addWidget(self.cbo_mode, stretch=1)

        self.txt_guide = QLineEdit()
        self.txt_guide.setPlaceholderText("Hướng dẫn ngón tay (VD: Nắm đấm, ngón cái xòe áp má ngoài)")

        self.txt_description = QLineEdit()
        self.txt_description.setPlaceholderText("Mô tả ý nghĩa cử chỉ hoặc hành động điều khiển")

        btn_add = QPushButton("➕ Thêm Cử Chỉ Vào Database")
        btn_add.setProperty("class", "PrimaryButton")
        btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add.clicked.connect(self.add_custom_gesture)

        layout.addWidget(self.txt_gesture_name)
        layout.addWidget(self.txt_display_name)
        layout.addLayout(mode_row)
        layout.addWidget(self.txt_guide)
        layout.addWidget(self.txt_description)
        layout.addWidget(btn_add)

        return card

    # ── 2. CARD: BỘ THU THẬP MẪU CHUỖI TRỰC TIẾP (45 FRAMES) ─────────────────

    def _build_card_recorder(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        lbl_t = QLabel("2. BỘ THU THẬP MẪU CHUỖI TRỰC TIẾP (45 FRAMES)")
        lbl_t.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;")
        layout.addWidget(lbl_t)

        # Hàng chọn cử chỉ
        sel_row = QHBoxLayout()
        lbl_sel = QLabel("Chọn cử chỉ cần quay:")
        lbl_sel.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        self.cbo_record_gesture = QComboBox()
        self.cbo_record_gesture.currentIndexChanged.connect(self._on_record_gesture_changed)
        sel_row.addWidget(lbl_sel)
        sel_row.addWidget(self.cbo_record_gesture, stretch=1)
        layout.addLayout(sel_row)

        # Thông tin mẫu hiện có & tác giả
        info_row = QHBoxLayout()
        self.lbl_record_count = QLabel("📦 Đã có: 0 mẫu")
        self.lbl_record_count.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px; font-weight: 700;")
        self.lbl_author_info = QLabel(f"👤 Người quay: {self.active_author}")
        self.lbl_author_info.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 12px; font-weight: 600;")
        info_row.addWidget(self.lbl_record_count)
        info_row.addStretch()
        info_row.addWidget(self.lbl_author_info)
        layout.addLayout(info_row)

        # Thanh tiến trình thu thập (0 / 45)
        self.progress_record = QProgressBar()
        self.progress_record.setMaximum(_TARGET_SEQ_LEN)
        self.progress_record.setValue(0)
        self.progress_record.setFixedHeight(12)
        self.progress_record.setTextVisible(True)
        self.progress_record.setFormat("%v / 45 frames")
        layout.addWidget(self.progress_record)

        # Nhãn trạng thái trực quan
        self.lbl_record_status = QLabel("Sẵn sàng. Đặt bàn tay trước webcam rồi bấm Bắt Đầu.")
        self.lbl_record_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-style: italic;")
        self.lbl_record_status.setWordWrap(True)
        layout.addWidget(self.lbl_record_status)

        # Nút Mở Phòng Thu (Camera Trực Tiếp) & Thu Nhanh & Hủy
        btn_box = QVBoxLayout()
        btn_box.setSpacing(8)

        self.btn_open_dialog = QPushButton("🎥 Mở Phòng Thu Mẫu (Xem Camera Trực Tiếp)")
        self.btn_open_dialog.setProperty("class", "PrimaryButton")
        self.btn_open_dialog.setMinimumHeight(40)
        self.btn_open_dialog.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_dialog.clicked.connect(self.open_recorder_dialog)
        btn_box.addWidget(self.btn_open_dialog)

        sub_btn_row = QHBoxLayout()
        self.btn_start_record = QPushButton("⚡ Thu Nhanh Tại Chỗ (45 frames)")
        self.btn_start_record.setProperty("class", "SecondaryButton")
        self.btn_start_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_record.clicked.connect(self.start_recording)

        self.btn_cancel_record = QPushButton("❌ Hủy")
        self.btn_cancel_record.setProperty("class", "SecondaryButton")
        self.btn_cancel_record.setEnabled(False)
        self.btn_cancel_record.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel_record.clicked.connect(self.cancel_recording)

        sub_btn_row.addWidget(self.btn_start_record, stretch=2)
        sub_btn_row.addWidget(self.btn_cancel_record, stretch=1)
        btn_box.addLayout(sub_btn_row)

        layout.addLayout(btn_box)

        return card

    # ── 3. CARD: HUẤN LUYỆN MÔ HÌNH HỌC SÂU (HOT-RELOAD) ─────────────────────

    def _build_card_training(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        lbl_t = QLabel("3. HUẤN LUYỆN MÔ HÌNH HỌC SÂU (HOT-RELOAD)")
        lbl_t.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;")
        layout.addWidget(lbl_t)

        lbl_desc = QLabel(
            "Gộp toàn bộ mẫu phân tán của bạn và bạn bè trong SQLite để train lại mạng Bi-LSTM 158D Context Gate. "
            "Sau khi train, mô hình mới tự động nạp mà không cần tắt App."
        )
        lbl_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Tham số cấu hình
        train_form = QHBoxLayout()
        train_form.addWidget(QLabel("Số Epochs:"))
        self.txt_epochs = QLineEdit("20")
        self.txt_epochs.setMaximumWidth(70)
        train_form.addWidget(self.txt_epochs)

        train_form.addWidget(QLabel("Batch Size:"))
        self.txt_batch = QLineEdit("32")
        self.txt_batch.setMaximumWidth(70)
        train_form.addWidget(self.txt_batch)
        train_form.addStretch()
        layout.addLayout(train_form)

        # Nút Huấn luyện & Hủy
        btn_box = QHBoxLayout()
        self.btn_train_ai = QPushButton("🚀 BẮT ĐẦU HUẤN LUYỆN & NẠP MODEL MỚI")
        self.btn_train_ai.setProperty("class", "PrimaryButton")
        self.btn_train_ai.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_train_ai.clicked.connect(self.run_training)

        self.btn_cancel_train = QPushButton("⏹ Hủy")
        self.btn_cancel_train.setProperty("class", "SecondaryButton")
        self.btn_cancel_train.setEnabled(False)
        self.btn_cancel_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel_train.clicked.connect(self.cancel_training)

        btn_box.addWidget(self.btn_train_ai, stretch=3)
        btn_box.addWidget(self.btn_cancel_train, stretch=1)
        layout.addLayout(btn_box)

        # Tiến trình train
        self.progress_train = QProgressBar()
        self.progress_train.setValue(0)
        self.progress_train.setFixedHeight(12)
        layout.addWidget(self.progress_train)

        self.lbl_train_status = QLabel("Trạng thái: Sẵn sàng huấn luyện")
        self.lbl_train_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self.lbl_train_status)

        # Hộp log kết quả train
        self.lbl_train_log = QLabel("Nhật ký: Chưa có lượt huấn luyện nào trong phiên hiện tại.")
        self.lbl_train_log.setStyleSheet(
            f"background-color: {DARK_BG_CARD}; border: 1px solid {DARK_BORDER};"
            f"border-radius: 6px; padding: 8px; color: {COLOR_GREEN}; font-family: 'Consolas', monospace; font-size: 11px;"
        )
        self.lbl_train_log.setWordWrap(True)
        layout.addWidget(self.lbl_train_log)

        return card

    # ── 4. CARD: QUẢN LÝ GÓI CHIA SẺ (ZIP) ────────────────────────────────────

    def _build_card_packages(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        lbl_t = QLabel("4. ĐỒNG BỘ & CHIA SẺ MẪU PHÂN TÁN (ZIP)")
        lbl_t.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;")
        layout.addWidget(lbl_t)

        lbl_desc = QLabel(
            "Đóng gói toàn bộ mẫu của bạn thành file .ZIP gửi cho bạn bè, "
            "hoặc nạp file .ZIP bạn bè gửi để hợp nhất cơ sở dữ liệu huấn luyện."
        )
        lbl_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("📦 Đóng Gói Của Tôi (.ZIP)")
        btn_export.setProperty("class", "SecondaryButton")
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export.clicked.connect(self.export_my_zip)

        btn_import = QPushButton("📥 Nhập Gói Bạn Bè (.ZIP)")
        btn_import.setProperty("class", "SecondaryButton")
        btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import.clicked.connect(self.import_friend_zip)

        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_import)
        layout.addLayout(btn_row)

        self.lbl_package_info = QLabel("Vị trí lưu mẫu phân tán: dataset/sequences/<cử_chỉ>/<tác_giả>_<thời_gian>.npz")
        self.lbl_package_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-style: italic;")
        layout.addWidget(self.lbl_package_info)

        return card

    # ── LOGIC 1: ĐĂNG KÝ CỬ CHỈ MỚI ──────────────────────────────────────────

    def add_custom_gesture(self) -> None:
        """Thêm cử chỉ mới vào SQLite và cập nhật dropdown."""
        name = self.txt_gesture_name.text().strip().upper()
        disp = self.txt_display_name.text().strip()
        guide = self.txt_guide.text().strip()
        desc = self.txt_description.text().strip() or "Cử chỉ tùy chỉnh thêm mới từ Studio"
        mode_id = self.cbo_mode.currentIndex() + 1

        if not name or not disp:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Tên mã và Tên hiển thị của cử chỉ!")
            return

        ok = self.db.add_gesture(
            name=name,
            display_name=disp,
            mode_id=mode_id,
            description=desc,
            finger_guide=guide,
            is_system=0,
        )
        if ok:
            QMessageBox.information(self, "Thành công", f"Đã thêm cử chỉ '{name}' vào cơ sở dữ liệu!")
            self.txt_gesture_name.clear()
            self.txt_display_name.clear()
            self.txt_guide.clear()
            self.txt_description.clear()
            self.refresh_gestures_dropdown()
            self.gesture_added.emit(name)
        else:
            QMessageBox.warning(self, "Trùng lặp", f"Cử chỉ có tên mã '{name}' đã tồn tại trong hệ thống!")

    def refresh_gestures_dropdown(self) -> None:
        """Cập nhật danh sách cử chỉ vào ComboBox thu thập mẫu."""
        current_sel = self.cbo_record_gesture.currentText()
        self.cbo_record_gesture.blockSignals(True)
        self.cbo_record_gesture.clear()

        try:
            gestures = self.db.get_all_gestures()
        except Exception:
            gestures = []

        for g in gestures:
            name = g.get("name", "")
            disp = g.get("display_name", "")
            self.cbo_record_gesture.addItem(f"{name} — {disp}", name)

        # Khôi phục lựa chọn cũ nếu có
        idx = self.cbo_record_gesture.findData(current_sel)
        if idx >= 0:
            self.cbo_record_gesture.setCurrentIndex(idx)

        self.cbo_record_gesture.blockSignals(False)
        self._on_record_gesture_changed()

    def _on_record_gesture_changed(self) -> None:
        """Cập nhật số lượng mẫu đã có của cử chỉ đang chọn."""
        gesture_name = self.get_selected_record_gesture()
        if not gesture_name:
            self.lbl_record_count.setText("📦 Đã có: 0 mẫu")
            return

        try:
            counts = self.db.get_gesture_sample_counts()
            cnt = counts.get(gesture_name, 0)
        except Exception:
            cnt = 0

        self.lbl_record_count.setText(f"📦 Đã có: {cnt} mẫu")

    def get_selected_record_gesture(self) -> str:
        """Lấy tên mã cử chỉ đang được chọn để quay."""
        data = self.cbo_record_gesture.currentData()
        if data:
            return str(data)
        # Fallback tách chuỗi
        txt = self.cbo_record_gesture.currentText()
        if "—" in txt:
            return txt.split("—")[0].strip()
        return txt.strip()

    # ── LOGIC 2: BỘ THU THẬP MẪU CHUỖI TRỰC TIẾP ────────────────────────────

    def open_recorder_dialog(self) -> None:
        """Mở Hộp thoại Thu Thập Mẫu Trực Quan chuyên dụng với màn hình Camera trực tiếp."""
        gesture_name = self.get_selected_record_gesture()
        if not gesture_name:
            QMessageBox.warning(self, "Chưa chọn cử chỉ", "Vui lòng chọn hoặc thêm một cử chỉ trước khi quay!")
            return

        g_info = self.db.get_gesture(gesture_name) or {
            "name": gesture_name,
            "display_name": gesture_name,
            "finger_guide": "Thực hiện cử chỉ trước camera",
            "mode_id": 2,
        }

        self.recorder_dialog = RecordingDialog(
            db=self.db,
            gesture_info=g_info,
            author=self.active_author,
            camera_worker=self.camera_worker,
            parent=self,
        )
        self.recorder_dialog.sample_recorded.connect(self._on_dialog_sample_recorded)
        self.recorder_dialog.sample_deleted.connect(self._on_dialog_sample_deleted)
        self.recorder_dialog.exec()

    def _on_dialog_sample_recorded(self, gesture_name: str, file_path: str) -> None:
        """Xử lý khi hộp thoại quay xong một mẫu."""
        self._on_record_gesture_changed()
        self.lbl_record_status.setText(f"✅ Vừa thu thập mẫu mới cho '{gesture_name}'!")
        self.lbl_record_status.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px; font-weight: 700;")
        self.sample_recorded.emit(gesture_name)

    def _on_dialog_sample_deleted(self, gesture_name: str, file_path: str) -> None:
        """Xử lý khi người dùng xóa mẫu vừa quay trong hộp thoại."""
        self._on_record_gesture_changed()
        self.lbl_record_status.setText(f"🗑️ Đã xóa mẫu bị lỗi của '{gesture_name}'!")
        self.lbl_record_status.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 12px; font-weight: 700;")

    def start_recording(self) -> None:
        """Bắt đầu chu trình đếm ngược 3... 2... 1... rồi thu thập 45 frames."""
        gesture_name = self.get_selected_record_gesture()
        if not gesture_name:
            QMessageBox.warning(self, "Chưa chọn cử chỉ", "Vui lòng chọn hoặc thêm một cử chỉ trước khi quay!")
            return

        if self.camera_worker:
            g_mode = 2
            g_info = self.db.get_gesture(gesture_name) if hasattr(self.db, "get_gesture") else None
            if g_info:
                g_mode = int(g_info.get("mode_id", 2))
            if hasattr(self.camera_worker, "set_recording_override"):
                self.camera_worker.set_recording_override(True, mode_id=g_mode)
            elif hasattr(self.camera_worker, "set_mouse_paused"):
                self.camera_worker.set_mouse_paused(True)

        self._is_recording = False
        self._record_buffer = []
        self.progress_record.setValue(0)
        self._countdown_val = 3

        self.btn_start_record.setEnabled(False)
        self.btn_cancel_record.setEnabled(True)
        self.cbo_record_gesture.setEnabled(False)

        self.lbl_record_status.setText(f"⏳ Chuẩn bị: Đếm ngược {self._countdown_val}...")
        self.lbl_record_status.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 13px; font-weight: 700;")
        self._countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        """Xử lý từng nhịp đếm ngược 3... 2... 1... -> BẮT ĐẦU."""
        self._countdown_val -= 1
        if self._countdown_val > 0:
            self.lbl_record_status.setText(f"⏳ Chuẩn bị: Đếm ngược {self._countdown_val}...")
        else:
            self._countdown_timer.stop()
            self._is_recording = True
            self.lbl_record_status.setText("🔴 ĐANG THU THẬP 45 FRAMES... Hãy thực hiện cử chỉ trước camera!")
            self.lbl_record_status.setStyleSheet(f"color: {COLOR_ROSE}; font-size: 13px; font-weight: 700;")

    def on_feature_frame(self, feat_158d: list) -> None:
        """Nhận vector 158D thời gian thực từ CameraWorker và gom vào buffer khi đang quay."""
        if not self._is_recording:
            return

        self._record_buffer.append(list(feat_158d))
        cur_len = len(self._record_buffer)
        self.progress_record.setValue(min(cur_len, _TARGET_SEQ_LEN))
        self.lbl_record_status.setText(f"🔴 Đang thu thập: {cur_len} / {_TARGET_SEQ_LEN} frames...")

        if cur_len >= _TARGET_SEQ_LEN:
            self._finish_recording()

    def feed_sample_sequence(self, sequence: np.ndarray) -> Path:
        """
        Nạp trực tiếp một chuỗi (N, 158) để lưu mà không cần camera (dành cho kiểm thử hoặc import).
        """
        gesture_name = self.get_selected_record_gesture() or "UNKNOWN_GESTURE"
        out_file = save_single_sample(
            sequence=sequence,
            gesture_name=gesture_name,
            author=self.active_author,
            db=self.db,
        )
        self._on_record_gesture_changed()
        self.sample_recorded.emit(gesture_name)
        return out_file

    def _finish_recording(self) -> None:
        """Hoàn tất thu thập đủ 45 frames, lưu file .npz và cập nhật SQLite."""
        self._is_recording = False
        self._countdown_timer.stop()

        if self.camera_worker:
            if hasattr(self.camera_worker, "set_recording_override"):
                self.camera_worker.set_recording_override(False)

        gesture_name = self.get_selected_record_gesture()
        arr = np.array(self._record_buffer[:_TARGET_SEQ_LEN], dtype=np.float32)

        try:
            saved_file = save_single_sample(
                sequence=arr,
                gesture_name=gesture_name,
                author=self.active_author,
                db=self.db,
            )
            self.lbl_record_status.setText(f"✅ Đã lưu mẫu thành công! File: {saved_file.name}")
            self.lbl_record_status.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px; font-weight: 700;")
            self.sample_recorded.emit(gesture_name)
        except Exception as e:
            self.lbl_record_status.setText(f"❌ Lỗi khi lưu file: {str(e)}")
            self.lbl_record_status.setStyleSheet(f"color: {COLOR_ROSE}; font-size: 12px;")

        self._reset_record_ui()
        self._on_record_gesture_changed()

    def cancel_recording(self) -> None:
        """Hủy quá trình quay mẫu đang diễn ra."""
        self._countdown_timer.stop()
        self._is_recording = False
        self._record_buffer = []

        if self.camera_worker:
            if hasattr(self.camera_worker, "set_recording_override"):
                self.camera_worker.set_recording_override(False)

        self.progress_record.setValue(0)
        self.lbl_record_status.setText("Đã hủy lượt quay mẫu.")
        self.lbl_record_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        self._reset_record_ui()

    def _reset_record_ui(self) -> None:
        """Khôi phục trạng thái nút bấm sau khi quay hoặc hủy."""
        self.btn_start_record.setEnabled(True)
        self.btn_cancel_record.setEnabled(False)
        self.cbo_record_gesture.setEnabled(True)

    # ── LOGIC 3: HUẤN LUYỆN MÔ HÌNH HỌC SÂU (HOT-RELOAD) ─────────────────────

    def run_training(self) -> None:
        """Khởi chạy huấn luyện mô hình qua QThread TrainWorker không đơ giao diện."""
        try:
            epochs = int(self.txt_epochs.text().strip() or "20")
            batch = int(self.txt_batch.text().strip() or "32")
        except ValueError:
            epochs = 20
            batch = 32

        self.btn_train_ai.setEnabled(False)
        self.btn_cancel_train.setEnabled(True)
        self.progress_train.setValue(0)
        self.lbl_train_status.setText("Trạng thái: Đang chuẩn bị dataset phân tán...")

        if self.camera_worker and hasattr(self.camera_worker, "set_mouse_paused"):
            self.camera_worker.set_mouse_paused(True)

        self.train_worker = TrainWorker(db=self.db, epochs=epochs, batch_size=batch, parent=self)
        self.train_worker.epoch_progress.connect(self._on_train_epoch_progress)
        self.train_worker.training_completed.connect(self._on_train_completed)
        self.train_worker.training_failed.connect(self._on_train_failed)
        self.train_worker.start()

    def cancel_training(self) -> None:
        """Hủy tiến trình huấn luyện."""
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.cancel()
            self.lbl_train_status.setText("Trạng thái: Đang dừng tiến trình huấn luyện...")

    def _on_train_epoch_progress(self, ep: int, total_epochs: int, loss: float, acc: float) -> None:
        pct = int((ep / max(1, total_epochs)) * 100)
        self.progress_train.setValue(pct)
        self.lbl_train_status.setText(f"Epoch {ep}/{total_epochs} | Loss: {loss:.4f} | Acc: {acc:.1f}%")

    def _on_train_completed(self, summary: Dict[str, Any]) -> None:
        self.btn_train_ai.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        self.progress_train.setValue(100)

        samples = summary.get("total_samples", 0)
        classes = summary.get("total_classes", 0)
        acc = summary.get("final_acc", 0.0)
        loss = summary.get("final_loss", 0.0)
        path = summary.get("model_path", "")

        self.lbl_train_status.setText(f"✅ Hoàn tất huấn luyện: {samples} mẫu, {classes} lớp, Acc: {acc}%")
        self.lbl_train_log.setText(
            f"KẾT QUẢ HUẤN LUYỆN:\n"
            f"  • Tổng số mẫu gộp: {samples} mẫu (45 frames/chuỗi)\n"
            f"  • Số lớp cử chỉ nhận diện: {classes} lớp\n"
            f"  • Độ chính xác (Accuracy): {acc:.2f}%\n"
            f"  • Tổn thất (Loss): {loss:.4f}\n"
            f"  • File mô hình: {path} (Đã nạp Hot-Reload thành công)"
        )

        QMessageBox.information(
            self,
            "Huấn luyện thành công!",
            f"Đã huấn luyện xong mạng nơ-ron Bi-LSTM 158D Context Gate!\n\n"
            f"• Số mẫu: {samples}\n"
            f"• Số cử chỉ: {classes}\n"
            f"• Độ chính xác: {acc:.2f}%\n\n"
            f"Mô hình mới đã được Hot-Reload trực tiếp vào luồng suy luận của App.",
        )
        self.training_finished.emit(summary)

    def _on_train_failed(self, error_msg: str) -> None:
        self.btn_train_ai.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        self.lbl_train_status.setText(f"❌ {error_msg}")
        self.lbl_train_log.setText(f"LỖI HUẤN LUYỆN:\n  {error_msg}")
        QMessageBox.warning(self, "Lỗi huấn luyện", error_msg)

    # ── LOGIC 4: XUẤT / NHẬP ZIP CHIA SẺ BẠN BÈ ──────────────────────────────

    def export_my_zip(self) -> None:
        """Xuất gói dữ liệu của người dùng hiện tại thành file ZIP."""
        author = self.active_author
        default_name = f"Dataset_{author}.zip"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu gói dữ liệu cử chỉ (.ZIP)",
            str(config.PROJECT_ROOT / default_name),
            "ZIP Files (*.zip)",
        )
        if not save_path:
            return

        try:
            n = export_contributor_zip(author, Path(save_path))
            QMessageBox.information(
                self,
                "Xuất gói thành công",
                f"Đã đóng gói thành công {n} mẫu do '{author}' thu thập vào file:\n{save_path}",
            )
        except Exception as e:
            QMessageBox.warning(self, "Lỗi xuất gói", f"Không thể xuất file ZIP: {str(e)}")

    def import_friend_zip(self) -> None:
        """Nhập file ZIP từ bạn bè vào hệ thống và đồng bộ SQLite."""
        open_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file gói dữ liệu của bạn bè (.ZIP)",
            str(config.PROJECT_ROOT),
            "ZIP Files (*.zip)",
        )
        if not open_path:
            return

        try:
            n = import_contributor_zip(Path(open_path), db=self.db)
            QMessageBox.information(
                self,
                "Nhập gói thành công",
                f"Đã nhập thành công {n} mẫu cử chỉ từ bạn bè vào hệ thống!\n"
                f"Cơ sở dữ liệu đã được tự động đồng bộ hóa.",
            )
            self.refresh_gestures_dropdown()
            self._on_record_gesture_changed()
        except Exception as e:
            QMessageBox.warning(self, "Lỗi nhập gói", f"Không thể nhập file ZIP: {str(e)}")
