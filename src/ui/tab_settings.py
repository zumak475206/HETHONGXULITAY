"""
src/ui/tab_settings.py
======================
Tab 4 — Cài Đặt Hệ Thống & Tinh Chỉnh Công Thái Học cho HETHONGXULITAY.

Cung cấp giao diện trực quan với các Cards độc lập:
  1. Công thái học con trỏ chuột (Mouse Sensitivity, Y-Gain gập cổ tay).
  2. Bộ lọc chống rung One Euro Filter (Beta tăng tốc, Min Cutoff chống rung micro-jitter).
  3. Ngưỡng thời gian kích hoạt cử chỉ (Hold Click Phải chữ V, Fist Grace Period mở tay, Two-Hand Switch).
  4. Cấu hình Camera & Bộ tăng cường hình ảnh CLAHE (Thiết bị, Độ phân giải HD/SD, CLAHE toggle).
  5. Hồ sơ người dùng & Khôi phục mặc định (Tên tác giả, Lưu cài đặt SQLite, Live Sync, Factory Reset).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import config
from src.database.db_manager import DatabaseManager
from src.ui.theme import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_ROSE,
    DARK_BG_CARD,
    DARK_BG_SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


# Hằng số mặc định chuẩn (Factory Defaults)
FACTORY_DEFAULTS = {
    "active_author": "Tôi",
    "mouse_sensitivity": 1.25,
    "mouse_y_gain": 1.55,
    "one_euro_beta": 0.040,
    "one_euro_min_cutoff": 1.0,
    "right_click_hold_time": 0.35,
    "fist_grace_time": 0.25,
    "two_hand_switch_time": 2.0,
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "clahe_enabled": True,
}


class SettingsPage(QWidget):
    """
    Trang Cài đặt hệ thống & Tinh chỉnh công thái học con trỏ chuột.
    Tự động lưu trữ xuống SQLite và đồng bộ thời gian thực (Live Sync) tới CameraWorker & MouseController.
    """

    settings_saved = pyqtSignal(dict)
    author_changed = pyqtSignal(str)
    factory_reset = pyqtSignal()

    def __init__(
        self,
        db: DatabaseManager,
        author: str = "Tôi",
        camera_worker: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.active_author = author
        self.camera_worker = camera_worker

        self._init_ui()
        self.load_settings()

    def set_camera_worker(self, camera_worker: Any) -> None:
        """Cập nhật tham chiếu CameraWorker và đồng bộ các cài đặt hiện hành."""
        self.camera_worker = camera_worker
        self._dispatch_live_settings()

    def _init_ui(self) -> None:
        """Khởi tạo toàn bộ bố cục trang cuộn mượt mà với các thẻ chức năng (Cards)."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header cố định phía trên
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(24, 18, 24, 12)
        header_layout.setSpacing(4)

        lbl_title = QLabel("CÀI ĐẶT HỆ THỐNG & TINH CHỈNH CÔNG THÁI HỌC")
        lbl_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: 800; letter-spacing: 0.5px;")
        header_layout.addWidget(lbl_title)

        lbl_sub = QLabel("Tùy biến độ nhạy con trỏ chuột, bộ lọc chống rung, thời gian kích hoạt cử chỉ và thiết bị camera.")
        lbl_sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-weight: 500;")
        header_layout.addWidget(lbl_sub)

        root_layout.addWidget(header_widget)

        # Vùng cuộn QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(24, 6, 24, 24)
        container_layout.setSpacing(16)

        # ── CARD 1: CÔNG THÁI HỌC CON TRỎ CHUỘT ──────────────────────────────
        card_mouse = self._create_card(
            title="🖱️ ĐIỀU KHIỂN CON TRỎ & GIA TỐC CỔ TAY (MOUSE ERGONOMICS)",
            desc="Điều chỉnh tốc độ di chuột và hệ số trợ lực trục dọc khi gập cổ tay xuống dưới.",
        )
        mouse_layout = QVBoxLayout(card_mouse)
        mouse_layout.setContentsMargins(20, 16, 20, 18)
        mouse_layout.setSpacing(14)

        # 1.1 Độ nhạy chuột (Sensitivity)
        row_sens = QVBoxLayout()
        row_sens.setSpacing(6)
        self.lbl_sens_val = QLabel("Độ nhạy con trỏ chuột (1.25x):")
        self.lbl_sens_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_sens.addWidget(self.lbl_sens_val)

        row_sens_ctrl = QHBoxLayout()
        lbl_sens_min = QLabel("0.80x")
        lbl_sens_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(80, 250)
        self.slider_sens.setValue(125)
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        lbl_sens_max = QLabel("2.50x")
        lbl_sens_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_sens_ctrl.addWidget(lbl_sens_min)
        row_sens_ctrl.addWidget(self.slider_sens, stretch=1)
        row_sens_ctrl.addWidget(lbl_sens_max)
        row_sens.addLayout(row_sens_ctrl)
        mouse_layout.addLayout(row_sens)

        # 1.2 Gia tốc phi tuyến trục Y khi gập cổ tay (Y-Gain)
        row_ygain = QVBoxLayout()
        row_ygain.setSpacing(6)
        self.lbl_ygain_val = QLabel("Gia tốc gập cổ tay trục Y (1.55x):")
        self.lbl_ygain_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_ygain.addWidget(self.lbl_ygain_val)

        row_ygain_ctrl = QHBoxLayout()
        lbl_ygain_min = QLabel("1.00x (Thẳng)")
        lbl_ygain_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_ygain = QSlider(Qt.Orientation.Horizontal)
        self.slider_ygain.setRange(100, 250)
        self.slider_ygain.setValue(155)
        self.slider_ygain.valueChanged.connect(self._on_ygain_changed)
        lbl_ygain_max = QLabel("2.50x (Siêu trợ lực)")
        lbl_ygain_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_ygain_ctrl.addWidget(lbl_ygain_min)
        row_ygain_ctrl.addWidget(self.slider_ygain, stretch=1)
        row_ygain_ctrl.addWidget(lbl_ygain_max)
        row_ygain.addLayout(row_ygain_ctrl)

        lbl_ygain_hint = QLabel("💡 Mẹo: Tăng hệ số này giúp ngón tay dễ dàng chạm đáy màn hình mà không cần ưỡn mỏi cổ tay.")
        lbl_ygain_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        row_ygain.addWidget(lbl_ygain_hint)
        mouse_layout.addLayout(row_ygain)

        container_layout.addWidget(card_mouse)

        # ── CARD 2: BỘ LỌC CHỐNG RUNG ONE EURO FILTER ────────────────────────
        card_filter = self._create_card(
            title="🎯 BỘ LỌC CHỐNG RUNG & TỐC ĐỘ PHẢN HỒI (ONE EURO FILTER)",
            desc="Lọc triệt để vi rung giật (micro-jitter) khi giữ yên và giảm tối đa độ trễ khi vung tay nhanh.",
        )
        filter_layout = QVBoxLayout(card_filter)
        filter_layout.setContentsMargins(20, 16, 20, 18)
        filter_layout.setSpacing(14)

        # 2.1 Hệ số Beta vận tốc (Beta gain)
        row_beta = QVBoxLayout()
        row_beta.setSpacing(6)
        self.lbl_beta_val = QLabel("Hệ số Beta tốc độ vung tay (0.040):")
        self.lbl_beta_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_beta.addWidget(self.lbl_beta_val)

        row_beta_ctrl = QHBoxLayout()
        lbl_beta_min = QLabel("0.005 (Đằm tay)")
        lbl_beta_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_beta = QSlider(Qt.Orientation.Horizontal)
        self.slider_beta.setRange(5, 150)
        self.slider_beta.setValue(40)
        self.slider_beta.valueChanged.connect(self._on_beta_changed)
        lbl_beta_max = QLabel("0.150 (Tức thì / 0ms)")
        lbl_beta_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_beta_ctrl.addWidget(lbl_beta_min)
        row_beta_ctrl.addWidget(self.slider_beta, stretch=1)
        row_beta_ctrl.addWidget(lbl_beta_max)
        row_beta.addLayout(row_beta_ctrl)

        lbl_beta_hint = QLabel("💡 Càng cao: Di chuyển nhanh càng triệt tiêu hoàn toàn độ trễ chuột.")
        lbl_beta_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        row_beta.addWidget(lbl_beta_hint)
        filter_layout.addLayout(row_beta)

        # 2.2 Tần số cắt tối thiểu (Min Cutoff)
        row_cutoff = QVBoxLayout()
        row_cutoff.setSpacing(6)
        self.lbl_cutoff_val = QLabel("Tần số cắt tối thiểu Min Cutoff (1.0 Hz):")
        self.lbl_cutoff_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_cutoff.addWidget(self.lbl_cutoff_val)

        row_cutoff_ctrl = QHBoxLayout()
        lbl_cutoff_min = QLabel("0.2 Hz (Vững thạch)")
        lbl_cutoff_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_min_cutoff = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_cutoff.setRange(2, 30)
        self.slider_min_cutoff.setValue(10)
        self.slider_min_cutoff.valueChanged.connect(self._on_cutoff_changed)
        lbl_cutoff_max = QLabel("3.0 Hz (Nhạy cao)")
        lbl_cutoff_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_cutoff_ctrl.addWidget(lbl_cutoff_min)
        row_cutoff_ctrl.addWidget(self.slider_min_cutoff, stretch=1)
        row_cutoff_ctrl.addWidget(lbl_cutoff_max)
        row_cutoff.addLayout(row_cutoff_ctrl)

        lbl_cutoff_hint = QLabel("💡 Càng thấp: Khi trỏ ngón dừng yên chuột sẽ đứng im hoàn toàn, loại bỏ rung cơ học từ nhịp thở.")
        lbl_cutoff_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        row_cutoff.addWidget(lbl_cutoff_hint)
        filter_layout.addLayout(row_cutoff)

        container_layout.addWidget(card_filter)

        # ── CARD 3: THỜI GIAN GIỮ & NGƯỠNG KÍCH HOẠT CỬ CHỈ ─────────────────
        card_gesture = self._create_card(
            title="⏱️ THỜI GIAN GIỮ & NGƯỠNG KÍCH HOẠT CỬ CHỈ",
            desc="Cấu hình độ trễ kích hoạt để loại bỏ hoàn toàn các cú kích hoạt ngoài ý muốn.",
        )
        gesture_layout = QVBoxLayout(card_gesture)
        gesture_layout.setContentsMargins(20, 16, 20, 18)
        gesture_layout.setSpacing(14)

        # 3.1 Thời gian giữ chữ V Click Phải
        row_rc = QVBoxLayout()
        row_rc.setSpacing(6)
        self.lbl_rc_val = QLabel("Thời gian sạc chữ V Click Phải (0.35s):")
        self.lbl_rc_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_rc.addWidget(self.lbl_rc_val)

        row_rc_ctrl = QHBoxLayout()
        lbl_rc_min = QLabel("0.15s")
        lbl_rc_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_right_click = QSlider(Qt.Orientation.Horizontal)
        self.slider_right_click.setRange(15, 80)
        self.slider_right_click.setValue(35)
        self.slider_right_click.valueChanged.connect(self._on_rc_changed)
        lbl_rc_max = QLabel("0.80s")
        lbl_rc_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_rc_ctrl.addWidget(lbl_rc_min)
        row_rc_ctrl.addWidget(self.slider_right_click, stretch=1)
        row_rc_ctrl.addWidget(lbl_rc_max)
        row_rc.addLayout(row_rc_ctrl)
        gesture_layout.addLayout(row_rc)

        # 3.2 Thời gian đệm mở Nắm Tay (Fist Grace Period)
        row_fist = QVBoxLayout()
        row_fist.setSpacing(6)
        self.lbl_fist_val = QLabel("Thời gian đệm mở Nắm Tay Fist Grace (0.25s):")
        self.lbl_fist_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_fist.addWidget(self.lbl_fist_val)

        row_fist_ctrl = QHBoxLayout()
        lbl_fist_min = QLabel("0.10s")
        lbl_fist_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_fist_grace = QSlider(Qt.Orientation.Horizontal)
        self.slider_fist_grace.setRange(10, 60)
        self.slider_fist_grace.setValue(25)
        self.slider_fist_grace.valueChanged.connect(self._on_fist_changed)
        lbl_fist_max = QLabel("0.60s")
        lbl_fist_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_fist_ctrl.addWidget(lbl_fist_min)
        row_fist_ctrl.addWidget(self.slider_fist_grace, stretch=1)
        row_fist_ctrl.addWidget(lbl_fist_max)
        row_fist.addLayout(row_fist_ctrl)

        lbl_fist_hint = QLabel("💡 Khoảng dừng sau khi vừa mở nắm tay ra, chống hiện tượng con trỏ nhảy vị trí đột ngột.")
        lbl_fist_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        row_fist.addWidget(lbl_fist_hint)
        gesture_layout.addLayout(row_fist)

        # 3.3 Thời gian giữ 2 tay đổi Mode
        row_mode_switch = QVBoxLayout()
        row_mode_switch.setSpacing(6)
        self.lbl_switch_val = QLabel("Thời gian giữ 2 tay để chuyển Mode (2.0s):")
        self.lbl_switch_val.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_mode_switch.addWidget(self.lbl_switch_val)

        row_ms_ctrl = QHBoxLayout()
        lbl_ms_min = QLabel("1.0s")
        lbl_ms_min.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.slider_switch_mode = QSlider(Qt.Orientation.Horizontal)
        self.slider_switch_mode.setRange(10, 40)
        self.slider_switch_mode.setValue(20)
        self.slider_switch_mode.valueChanged.connect(self._on_switch_changed)
        lbl_ms_max = QLabel("4.0s")
        lbl_ms_max.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row_ms_ctrl.addWidget(lbl_ms_min)
        row_ms_ctrl.addWidget(self.slider_switch_mode, stretch=1)
        row_ms_ctrl.addWidget(lbl_ms_max)
        row_mode_switch.addLayout(row_ms_ctrl)
        gesture_layout.addLayout(row_mode_switch)

        container_layout.addWidget(card_gesture)

        # ── CARD 4: THIẾT BỊ CAMERA & TỐI ƯU HÌNH ẢNH ─────────────────────────
        card_cam = self._create_card(
            title="📷 THIẾT BỊ WEBCAM & BỘ TĂNG CƯỜNG HÌNH ẢNH",
            desc="Lựa chọn cảm biến camera và kích hoạt xử lý bù sáng thích ứng CLAHE.",
        )
        cam_layout = QVBoxLayout(card_cam)
        cam_layout.setContentsMargins(20, 16, 20, 18)
        cam_layout.setSpacing(14)

        # 4.1 Cổng thiết bị Camera Index
        row_cam_idx = QHBoxLayout()
        lbl_cam_idx = QLabel("Cổng thiết bị Camera (Device Index):")
        lbl_cam_idx.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_cam_idx.addWidget(lbl_cam_idx)

        self.cbo_camera_idx = QComboBox()
        self.cbo_camera_idx.addItems([
            "Camera 0 (Webcam tích hợp / Mặc định)",
            "Camera 1 (Webcam cắm ngoài USB 1)",
            "Camera 2 (Webcam cắm ngoài USB 2)",
        ])
        self.cbo_camera_idx.setMinimumWidth(320)
        row_cam_idx.addWidget(self.cbo_camera_idx, stretch=1)
        cam_layout.addLayout(row_cam_idx)

        # 4.2 Độ phân giải Camera
        row_res = QHBoxLayout()
        lbl_res = QLabel("Độ phân giải khung hình mặc định:")
        lbl_res.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_res.addWidget(lbl_res)

        self.cbo_res = QComboBox()
        self.cbo_res.addItems([
            "1280x720 (HD Sắc Nét - Khuyến nghị cho Zero-Lag)",
            "640x480 (Tiêu Chuẩn SD - Tiết kiệm CPU / Máy yếu)",
        ])
        self.cbo_res.setMinimumWidth(320)
        row_res.addWidget(self.cbo_res, stretch=1)
        cam_layout.addLayout(row_res)

        # 4.3 Checkbox CLAHE
        self.chk_clahe = QCheckBox("Bật bộ tăng cường ảnh chống chói & bù sáng CLAHE (Dynamic Gamma + Anti-Glare)")
        self.chk_clahe.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600; padding: 4px 0;")
        self.chk_clahe.setChecked(True)
        cam_layout.addWidget(self.chk_clahe)

        container_layout.addWidget(card_cam)

        # ── CARD 5: HỒ SƠ TÁC GIẢ & THAO TÁC HỆ THỐNG ─────────────────────────
        card_profile = self._create_card(
            title="👤 HỒ SƠ TÁC GIẢ & THAO TÁC HỆ THỐNG",
            desc="Lưu các thông số vào cơ sở dữ liệu SQLite và đồng bộ ngay lập tức tới luồng điều khiển.",
        )
        profile_layout = QVBoxLayout(card_profile)
        profile_layout.setContentsMargins(20, 16, 20, 18)
        profile_layout.setSpacing(14)

        # 5.1 Tên người dùng / Tác giả
        row_author = QHBoxLayout()
        lbl_author = QLabel("Tên tác giả đóng góp:")
        lbl_author.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        row_author.addWidget(lbl_author)

        self.txt_author = QLineEdit(self.active_author)
        self.txt_author.setPlaceholderText("Nhập tên bạn (VD: Kỳ, Huy, Lan...)")
        row_author.addWidget(self.txt_author, stretch=1)
        profile_layout.addLayout(row_author)

        # 5.2 Hàng nút thao tác
        row_btns = QHBoxLayout()
        row_btns.setSpacing(12)

        self.btn_save = QPushButton("💾 Lưu Cài Đặt & Đồng Bộ Thời Gian Thực")
        self.btn_save.setProperty("class", "PrimaryButton")
        self.btn_save.setMinimumHeight(42)
        self.btn_save.clicked.connect(self.save_settings)
        row_btns.addWidget(self.btn_save, stretch=2)

        self.btn_reset = QPushButton("🔄 Khôi Phục Mặc Định (Factory Reset)")
        self.btn_reset.setProperty("class", "DangerButton")
        self.btn_reset.setMinimumHeight(42)
        self.btn_reset.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(244, 63, 94, 0.15);
                color: {COLOR_ROSE};
                border: 1px solid rgba(244, 63, 94, 0.40);
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: rgba(244, 63, 94, 0.28);
                border-color: {COLOR_ROSE};
            }}
        """)
        self.btn_reset.clicked.connect(self.reset_defaults)
        row_btns.addWidget(self.btn_reset, stretch=1)

        profile_layout.addLayout(row_btns)

        # Trạng thái thông báo lưu
        self.lbl_save_status = QLabel("")
        self.lbl_save_status.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px; font-weight: 600;")
        self.lbl_save_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        profile_layout.addWidget(self.lbl_save_status)

        container_layout.addWidget(card_profile)

        # Khoảng co giãn cuối container
        container_layout.addStretch()

        scroll_area.setWidget(container)
        root_layout.addWidget(scroll_area)

    def _create_card(self, title: str, desc: str) -> QFrame:
        """Tạo khung Card tiêu chuẩn với phong cách Modern Dark."""
        card = QFrame()
        card.setProperty("class", "CardFrame")
        card.setStyleSheet(f"""
            QFrame[class="CardFrame"] {{
                background-color: {DARK_BG_CARD};
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
            }}
        """)
        # Card header layout
        outer_box = QVBoxLayout()
        outer_box.setContentsMargins(0, 0, 0, 0)
        outer_box.setSpacing(0)

        hdr_frame = QFrame()
        hdr_frame.setStyleSheet(f"""
            background-color: {DARK_BG_SURFACE};
            border-top-left-radius: 11px;
            border-top-right-radius: 11px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding: 8px 16px;
        """)
        hdr_layout = QVBoxLayout(hdr_frame)
        hdr_layout.setContentsMargins(16, 12, 16, 12)
        hdr_layout.setSpacing(3)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 13px; font-weight: 700; letter-spacing: 0.3px;")
        hdr_layout.addWidget(lbl_t)

        lbl_d = QLabel(desc)
        lbl_d.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        hdr_layout.addWidget(lbl_d)

        # Chèn header vào card trước khi trả về
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(hdr_frame)
        return card

    # ── XỬ LÝ SỰ KIỆN SLIDER (SMOOTH DISPLAY UPDATE) ─────────────────────────

    def _on_sens_changed(self, val: int) -> None:
        sens = val / 100.0
        self.lbl_sens_val.setText(f"Độ nhạy con trỏ chuột ({sens:.2f}x):")

    def _on_ygain_changed(self, val: int) -> None:
        gain = val / 100.0
        self.lbl_ygain_val.setText(f"Gia tốc gập cổ tay trục Y ({gain:.2f}x):")

    def _on_beta_changed(self, val: int) -> None:
        beta = val / 1000.0
        self.lbl_beta_val.setText(f"Hệ số Beta tốc độ vung tay ({beta:.3f}):")

    def _on_cutoff_changed(self, val: int) -> None:
        cutoff = val / 10.0
        self.lbl_cutoff_val.setText(f"Tần số cắt tối thiểu Min Cutoff ({cutoff:.1f} Hz):")

    def _on_rc_changed(self, val: int) -> None:
        rc = val / 100.0
        self.lbl_rc_val.setText(f"Thời gian sạc chữ V Click Phải ({rc:.2f}s):")

    def _on_fist_changed(self, val: int) -> None:
        fist = val / 100.0
        self.lbl_fist_val.setText(f"Thời gian đệm mở Nắm Tay Fist Grace ({fist:.2f}s):")

    def _on_switch_changed(self, val: int) -> None:
        sw = val / 10.0
        self.lbl_switch_val.setText(f"Thời gian giữ 2 tay để chuyển Mode ({sw:.1f}s):")

    # ── ĐỌC VÀ LƯU CÀI ĐẶT (SQLITE & LIVE SYNC) ──────────────────────────────

    def load_settings(self) -> None:
        """Đọc cài đặt hiện có từ SQLite Database và nạp lên các Widget."""
        try:
            # 1. Author
            saved_author = self.db.get_setting("active_author", "")
            if not saved_author or (self.active_author and self.active_author != "Tôi" and saved_author == "Tôi"):
                saved_author = self.active_author or FACTORY_DEFAULTS["active_author"]
            self.txt_author.setText(saved_author)
            self.active_author = saved_author

            # 2. Sensitivity
            sens_val = float(self.db.get_setting("mouse_sensitivity", str(FACTORY_DEFAULTS["mouse_sensitivity"])))
            self.slider_sens.setValue(int(sens_val * 100))
            self._on_sens_changed(int(sens_val * 100))

            # 3. Y-Gain
            ygain_val = float(self.db.get_setting("mouse_y_gain", str(FACTORY_DEFAULTS["mouse_y_gain"])))
            self.slider_ygain.setValue(int(ygain_val * 100))
            self._on_ygain_changed(int(ygain_val * 100))

            # 4. Beta
            beta_val = float(self.db.get_setting("one_euro_beta", str(FACTORY_DEFAULTS["one_euro_beta"])))
            self.slider_beta.setValue(int(beta_val * 1000))
            self._on_beta_changed(int(beta_val * 1000))

            # 5. Min Cutoff
            cutoff_val = float(self.db.get_setting("one_euro_min_cutoff", str(FACTORY_DEFAULTS["one_euro_min_cutoff"])))
            self.slider_min_cutoff.setValue(int(cutoff_val * 10))
            self._on_cutoff_changed(int(cutoff_val * 10))

            # 6. Right click hold time
            rc_val = float(self.db.get_setting("right_click_hold_time", str(FACTORY_DEFAULTS["right_click_hold_time"])))
            self.slider_right_click.setValue(int(rc_val * 100))
            self._on_rc_changed(int(rc_val * 100))

            # 7. Fist grace time
            fist_val = float(self.db.get_setting("fist_grace_time", str(FACTORY_DEFAULTS["fist_grace_time"])))
            self.slider_fist_grace.setValue(int(fist_val * 100))
            self._on_fist_changed(int(fist_val * 100))

            # 8. Two hand switch time
            switch_val = float(self.db.get_setting("two_hand_switch_time", str(FACTORY_DEFAULTS["two_hand_switch_time"])))
            self.slider_switch_mode.setValue(int(switch_val * 10))
            self._on_switch_changed(int(switch_val * 10))

            # 9. Camera Index
            cam_idx = int(self.db.get_setting("camera_index", str(FACTORY_DEFAULTS["camera_index"])))
            self.cbo_camera_idx.setCurrentIndex(max(0, min(cam_idx, self.cbo_camera_idx.count() - 1)))

            # 10. Resolution
            cam_w = self.db.get_setting("camera_width", str(FACTORY_DEFAULTS["camera_width"]))
            self.cbo_res.setCurrentIndex(0 if cam_w == "1280" else 1)

            # 11. CLAHE
            clahe_on = self.db.get_setting("clahe_enabled", "1" if FACTORY_DEFAULTS["clahe_enabled"] else "0")
            self.chk_clahe.setChecked(clahe_on in ["1", "true", "True"])

            self._dispatch_live_settings()
        except Exception as e:
            print(f"[SettingsPage] Lỗi load settings: {e}")

    def get_current_settings(self) -> Dict[str, Any]:
        """Lấy toàn bộ thông số cài đặt hiện hành từ giao diện."""
        return {
            "active_author": self.txt_author.text().strip() or "Tôi",
            "mouse_sensitivity": round(self.slider_sens.value() / 100.0, 2),
            "mouse_y_gain": round(self.slider_ygain.value() / 100.0, 2),
            "one_euro_beta": round(self.slider_beta.value() / 1000.0, 4),
            "one_euro_min_cutoff": round(self.slider_min_cutoff.value() / 10.0, 2),
            "right_click_hold_time": round(self.slider_right_click.value() / 100.0, 2),
            "fist_grace_time": round(self.slider_fist_grace.value() / 100.0, 2),
            "two_hand_switch_time": round(self.slider_switch_mode.value() / 10.0, 2),
            "camera_index": self.cbo_camera_idx.currentIndex(),
            "camera_width": 1280 if self.cbo_res.currentIndex() == 0 else 640,
            "camera_height": 720 if self.cbo_res.currentIndex() == 0 else 480,
            "clahe_enabled": self.chk_clahe.isChecked(),
        }

    def save_settings(self) -> None:
        """Lưu toàn bộ thông số vào SQLite và gửi tín hiệu đồng bộ runtime."""
        s = self.get_current_settings()

        self.db.set_setting("active_author", s["active_author"], "Tên người dùng đóng góp mẫu")
        self.db.set_setting("mouse_sensitivity", f"{s['mouse_sensitivity']:.2f}", "Độ nhạy con trỏ chuột Mode 1")
        self.db.set_setting("mouse_y_gain", f"{s['mouse_y_gain']:.2f}", "Gia tốc phi tuyến trục Y khi gập cổ tay")
        self.db.set_setting("one_euro_beta", f"{s['one_euro_beta']:.4f}", "Hệ số Beta tốc độ vung tay One Euro Filter")
        self.db.set_setting("one_euro_min_cutoff", f"{s['one_euro_min_cutoff']:.2f}", "Tần số cắt tối thiểu chống rung")
        self.db.set_setting("right_click_hold_time", f"{s['right_click_hold_time']:.2f}", "Thời gian giữ sạc chữ V Click Phải")
        self.db.set_setting("fist_grace_time", f"{s['fist_grace_time']:.2f}", "Thời gian đệm mở Nắm Tay")
        self.db.set_setting("two_hand_switch_time", f"{s['two_hand_switch_time']:.2f}", "Thời gian giữ 2 tay để chuyển Mode")
        self.db.set_setting("camera_index", str(s["camera_index"]), "Chỉ số thiết bị camera")
        self.db.set_setting("camera_width", str(s["camera_width"]), "Độ rộng khung hình camera")
        self.db.set_setting("camera_height", str(s["camera_height"]), "Độ cao khung hình camera")
        self.db.set_setting("clahe_enabled", "1" if s["clahe_enabled"] else "0", "Trạng thái bật/tắt bộ lọc CLAHE")

        self.active_author = s["active_author"]
        self.author_changed.emit(self.active_author)
        self.settings_saved.emit(s)

        # Cập nhật trực tiếp tới CameraWorker & MouseController
        self._dispatch_live_settings()

        self.lbl_save_status.setText("✓ Đã lưu cài đặt và đồng bộ thời gian thực thành công!")
        try:
            QMessageBox.information(
                self,
                "Đã lưu cài đặt",
                "Toàn bộ cài đặt hệ thống đã được lưu vào cơ sở dữ liệu SQLite\n"
                "và áp dụng trực tiếp vào luồng nhận diện mà không cần khởi động lại!",
            )
        except Exception:
            pass

    def reset_defaults(self) -> None:
        """Khôi phục toàn bộ cài đặt về trạng thái chuẩn gốc của nhà sản xuất (Factory Reset)."""
        try:
            ret = QMessageBox.question(
                self,
                "Xác nhận Khôi phục Mặc định",
                "Bạn có chắc chắn muốn khôi phục toàn bộ độ nhạy chuột, bộ lọc chống rung,\n"
                "và thời gian cử chỉ về giá trị gốc của nhà sản xuất?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
        except Exception:
            pass

        # Nạp giá trị mặc định lên controls
        self.slider_sens.setValue(int(FACTORY_DEFAULTS["mouse_sensitivity"] * 100))
        self.slider_ygain.setValue(int(FACTORY_DEFAULTS["mouse_y_gain"] * 100))
        self.slider_beta.setValue(int(FACTORY_DEFAULTS["one_euro_beta"] * 1000))
        self.slider_min_cutoff.setValue(int(FACTORY_DEFAULTS["one_euro_min_cutoff"] * 10))
        self.slider_right_click.setValue(int(FACTORY_DEFAULTS["right_click_hold_time"] * 100))
        self.slider_fist_grace.setValue(int(FACTORY_DEFAULTS["fist_grace_time"] * 100))
        self.slider_switch_mode.setValue(int(FACTORY_DEFAULTS["two_hand_switch_time"] * 10))
        self.cbo_camera_idx.setCurrentIndex(FACTORY_DEFAULTS["camera_index"])
        self.cbo_res.setCurrentIndex(0)  # 1280x720
        self.chk_clahe.setChecked(FACTORY_DEFAULTS["clahe_enabled"])

        # Lưu tự động
        self.save_settings()
        self.factory_reset.emit()
        self.lbl_save_status.setText("✓ Đã khôi phục cài đặt gốc mặc định!")

    def _dispatch_live_settings(self) -> None:
        """Đồng bộ ngay tức thì tới CameraWorker đang chạy."""
        if not self.camera_worker:
            return

        s = self.get_current_settings()
        if hasattr(self.camera_worker, "update_mouse_settings"):
            self.camera_worker.update_mouse_settings(
                sensitivity=s["mouse_sensitivity"],
                y_gain=s["mouse_y_gain"],
                beta=s["one_euro_beta"],
                min_cutoff=s["one_euro_min_cutoff"],
                right_click_hold_time=s["right_click_hold_time"],
                fist_grace_time=s["fist_grace_time"],
            )

        if hasattr(self.camera_worker, "set_clahe_enabled"):
            self.camera_worker.set_clahe_enabled(s["clahe_enabled"])

        if hasattr(self.camera_worker, "set_two_hand_switch_time"):
            self.camera_worker.set_two_hand_switch_time(s["two_hand_switch_time"])
