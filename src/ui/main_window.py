"""
src/ui/main_window.py
=====================
Cửa sổ chính của ứng dụng Desktop HETHONGXULITAY (PyQt6).
Bảng điều khiển quản trị trung tâm: Sidebar Navigation, Live Camera Canvas,
Dải nút chọn Mode phát sáng, Dashboard SQLite, Studio Training và Cài đặt.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont, QColor
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QFrame,
    QGridLayout,
    QScrollArea,
    QLineEdit,
    QComboBox,
    QSlider,
    QProgressBar,
    QMessageBox,
)

import config
from src.database.db_manager import DatabaseManager
from src.database.dataset_merger import load_merged_dataset
from src.ui.pip_widget import PiPWidget
from src.ui.workers import CameraWorker, TrainWorker
from src.ui.tab_library import LibraryPage
from src.ui.tab_studio import StudioPage
from src.ui.tab_settings import SettingsPage
from src.ui.theme import (
    MODERN_DARK_QSS,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_ROSE,
    COLOR_AMBER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)


class MainWindow(QMainWindow):
    """Cửa sổ ứng dụng Desktop quản trị HETHONGXULITAY."""

    def __init__(self, db: Optional[DatabaseManager] = None, start_camera: bool = True, parent=None):
        super().__init__(parent)
        self.db = db if db else DatabaseManager()
        self.start_camera = start_camera

        self.setWindowTitle("HETHONGXULITAY — AI Hand Gesture Studio & Control Center")
        self.setMinimumSize(1240, 780)
        self.resize(1320, 840)

        # Trạng thái hiện tại của ứng dụng
        self.current_mode_name = self.db.get_setting("current_mode", "MODE1_MOUSE")
        self.active_author = self.db.get_setting("active_author", "Tôi")
        self.is_paused = False

        # Khởi tạo PiP Widget
        self.pip_widget = PiPWidget()
        self.pip_widget.restore_requested.connect(self._restore_from_pip)

        self._init_ui()
        self.apply_theme()

        # Nạp dữ liệu ban đầu
        self.refresh_stats()

        # Khởi tạo CameraWorker xử lý nền
        self.camera_worker = CameraWorker(initial_mode=self.current_mode_name, parent=self)
        self.camera_worker.frame_ready.connect(self._on_worker_frame_ready)
        self.camera_worker.metrics_ready.connect(self._on_worker_metrics_ready)
        self.camera_worker.mode_switched.connect(self._on_worker_mode_switched)
        self.camera_worker.error_occurred.connect(self._on_worker_error)

        if self.start_camera:
            self.camera_worker.start()

        if hasattr(self, "studio_page") and self.studio_page:
            self.studio_page.connect_camera_worker(self.camera_worker)

        if hasattr(self, "settings_page") and self.settings_page:
            self.settings_page.set_camera_worker(self.camera_worker)

    def _init_ui(self) -> None:
        """Khởi tạo toàn bộ cấu trúc giao diện chính."""
        central_widget = QWidget(self)
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. SIDEBAR BÊN TRÁI
        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        # 2. KHU VỰC CHÍNH BÊN PHẢI (Header + Stacked Pages)
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = self._build_header()
        main_layout.addWidget(header)

        # Stacked pages
        self.stacked_widget = QStackedWidget()
        self.page_home = self._build_page_home()
        self.page_library = self._build_page_library()
        self.page_studio = self._build_page_studio()
        self.page_settings = self._build_page_settings()

        self.stacked_widget.addWidget(self.page_home)      # Index 0
        self.stacked_widget.addWidget(self.page_library)   # Index 1
        self.stacked_widget.addWidget(self.page_studio)    # Index 2
        self.stacked_widget.addWidget(self.page_settings)  # Index 3

        main_layout.addWidget(self.stacked_widget, stretch=1)
        root_layout.addWidget(main_area, stretch=1)

        # Mặc định mở Trang Chủ
        self._switch_page(0)

    # ── 1. SIDEBAR NAVIGATION ────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        """Xây dựng thanh điều hướng Sidebar bên trái."""
        sidebar = QWidget()
        sidebar.setObjectName("SidebarContainer")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 16)
        layout.setSpacing(6)

        # Logo & App Title
        logo_box = QVBoxLayout()
        logo_box.setContentsMargins(10, 0, 10, 15)
        logo_box.setSpacing(2)

        lbl_logo = QLabel("⚡ HETHONGXULITAY")
        lbl_logo.setObjectName("SidebarLogoTitle")
        lbl_sub = QLabel("AI Hand Gesture Studio v2.0")
        lbl_sub.setObjectName("SidebarSubtitle")

        logo_box.addWidget(lbl_logo)
        logo_box.addWidget(lbl_sub)
        layout.addLayout(logo_box)

        # Phân cách mỏng
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #1e293b; max-height: 1px; margin-bottom: 8px;")
        layout.addWidget(line)

        # Danh sách nút bấm điều hướng
        self.nav_buttons: List[QPushButton] = []
        nav_items = [
            ("🎥  Trang Chủ Live", 0),
            ("📊  Kho Cử Chỉ & Thống Kê", 1),
            ("🔬  Studio Training AI", 2),
            ("⚙️  Cài Đặt Hệ Thống", 3),
        ]

        for title, idx in nav_items:
            btn = QPushButton(title)
            btn.setProperty("class", "SidebarNavButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, i=idx: self._switch_page(i))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()

        # Phần chân Sidebar: Thông tin tác giả & Engine
        user_card = QFrame()
        user_card.setProperty("class", "CardFrame")
        user_layout = QVBoxLayout(user_card)
        user_layout.setContentsMargins(12, 12, 12, 12)
        user_layout.setSpacing(4)

        self.lbl_author_display = QLabel(f"👤 Tác giả: {self.active_author}")
        self.lbl_author_display.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: 700; font-size: 12px;")
        lbl_engine = QLabel("⚡ Dual-Pipeline Zero-Lag")
        lbl_engine.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 11px;")
        lbl_version = QLabel("Bản thử nghiệm CP8")
        lbl_version.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")

        user_layout.addWidget(self.lbl_author_display)
        user_layout.addWidget(lbl_engine)
        user_layout.addWidget(lbl_version)
        layout.addWidget(user_card)

        return sidebar

    # ── 2. HEADER BAR ────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        """Xây dựng thanh Header trên cùng hiển thị trạng thái hệ thống."""
        header = QWidget()
        header.setObjectName("HeaderContainer")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(15)

        # Tiêu đề trang hiện tại
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self.lbl_header_title = QLabel("TRANG CHỦ / LIVE CAMERA FEED")
        self.lbl_header_title.setObjectName("HeaderTitle")
        self.lbl_header_sub = QLabel("Theo dõi thời gian thực & điều khiển máy tính bằng cử chỉ tay")
        self.lbl_header_sub.setObjectName("HeaderSubtitle")
        title_box.addWidget(self.lbl_header_title)
        title_box.addWidget(self.lbl_header_sub)
        layout.addLayout(title_box)

        layout.addStretch()

        # Pill trạng thái hệ thống (System Status Pill)
        status_pill = QFrame()
        status_pill.setObjectName("StatusPill")
        pill_layout = QHBoxLayout(status_pill)
        pill_layout.setContentsMargins(10, 4, 12, 4)
        pill_layout.setSpacing(8)

        dot = QLabel()
        dot.setObjectName("StatusDotGreen")
        self.lbl_status = QLabel("ThreadedCamera: SẴN SÀNG | 0ms Lag")
        self.lbl_status.setObjectName("StatusText")

        pill_layout.addWidget(dot)
        pill_layout.addWidget(self.lbl_status)
        layout.addWidget(status_pill)

        # Badge chế độ hiện hành
        self.lbl_mode_badge = QLabel("MODE 1: CHUỘT")
        self.lbl_mode_badge.setStyleSheet(f"""
            background-color: rgba(16, 185, 129, 0.15);
            color: {COLOR_GREEN};
            border: 1px solid {COLOR_GREEN};
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
        """)
        layout.addWidget(self.lbl_mode_badge)

        return header

    # ── 3. PAGE 0: TRANG CHỦ / LIVE CAMERA HUB ───────────────────────────────

    def _build_page_home(self) -> QWidget:
        """Trang chủ: Khung hình Camera Feed, Bảng đo nhanh & Dải nút chuyển Mode."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Vùng giữa: Camera Viewport (bên trái) + Bảng trạng thái nhanh (bên phải)
        center_row = QHBoxLayout()
        center_row.setSpacing(16)

        # Canvas Camera (Placeholder cho CP9)
        self.camera_canvas = QFrame()
        self.camera_canvas.setObjectName("CameraCanvasFrame")
        self.camera_canvas.setStyleSheet("""
            background-color: #050811;
            border: 2px dashed #1e293b;
            border-radius: 12px;
        """)
        canvas_layout = QVBoxLayout(self.camera_canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self.lbl_cam_preview = QLabel()
        self.lbl_cam_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cam_preview.setStyleSheet("background-color: transparent;")
        self.lbl_cam_preview.setMinimumSize(320, 240)
        self.lbl_cam_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        # Placeholder text trước khi camera kết nối
        self._show_camera_placeholder()
        canvas_layout.addWidget(self.lbl_cam_preview)

        center_row.addWidget(self.camera_canvas, stretch=7)

        # Bảng thông số đo đạc nhanh bên phải (Side HUD Card)
        side_card = QFrame()
        side_card.setProperty("class", "CardFrame")
        side_card.setStyleSheet(f"background-color: #11192b; border: 1px solid #1e293b; border-radius: 12px; max-width: 320px;")
        side_layout = QVBoxLayout(side_card)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        lbl_side_title = QLabel("BẢNG ĐO THỜI GIAN THỰC")
        lbl_side_title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 13px; font-weight: 700;")
        side_layout.addWidget(lbl_side_title)

        # Các chỉ số
        self.lbl_hud_fps = QLabel("Tốc độ khung hình: 60.0 FPS")
        self.lbl_hud_action = QLabel("Hành động: Di chuot (0ms)")
        self.lbl_hud_pipeline = QLabel("Đường truyền: Dual-Pipeline (Raw + OneEuro)")
        self.lbl_hud_guard = QLabel("Bảo vệ sinh học: HOẠT ĐỘNG (6 lớp)")

        for lbl in [self.lbl_hud_fps, self.lbl_hud_action, self.lbl_hud_pipeline, self.lbl_hud_guard]:
            lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; padding: 4px 0px;")
            side_layout.addWidget(lbl)

        side_layout.addStretch()

        # Nút bật chế độ Cửa sổ Mini ghim góc (Picture-in-Picture)
        self.btn_pip = QPushButton("🗗  Thu Nhỏ Mini (PiP)")
        self.btn_pip.setProperty("class", "SecondaryButton")
        self.btn_pip.setToolTip("Thu nhỏ camera thành ô vuông ghim góc màn hình để tiện lướt web/làm việc")
        self.btn_pip.clicked.connect(self._toggle_pip)
        side_layout.addWidget(self.btn_pip)

        center_row.addWidget(side_card, stretch=3)
        layout.addLayout(center_row, stretch=1)

        # Dải nút chọn Mode nhỏ bên dưới Camera (Mode Strip)
        mode_strip = QFrame()
        mode_strip.setProperty("class", "CardFrame")
        mode_strip.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 24px; padding: 6px 12px;")
        strip_layout = QHBoxLayout(mode_strip)
        strip_layout.setContentsMargins(10, 4, 10, 4)
        strip_layout.setSpacing(8)

        lbl_strip_hint = QLabel("CHỌN CHẾ ĐỘ:")
        lbl_strip_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; margin-right: 8px;")
        strip_layout.addWidget(lbl_strip_hint)

        self.btn_mode1 = QPushButton("● MODE 1: CHUỘT (0ms)")
        self.btn_mode2 = QPushButton("MODE 2: TRÌNH CHIẾU / MEDIA")
        self.btn_mode3 = QPushButton("MODE 3: KÝ HIỆU NNKH")
        self.btn_standby = QPushButton("⏸ TẠM KHÓA (STANDBY)")

        self.mode_buttons = {
            "MODE1_MOUSE": self.btn_mode1,
            "MODE2_GESTURES": self.btn_mode2,
            "MODE3_SIGN": self.btn_mode3,
        }

        for mode_key, btn in self.mode_buttons.items():
            btn.setProperty("class", "ModeStripButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode_key: self._on_mode_button_clicked(m))
            strip_layout.addWidget(btn)

        self.btn_standby.setProperty("class", "DangerButton")
        self.btn_standby.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_standby.setStyleSheet("border-radius: 20px; padding: 6px 16px;")
        self.btn_standby.clicked.connect(self._toggle_standby)
        strip_layout.addWidget(self.btn_standby)

        strip_layout.addStretch()
        lbl_guide = QLabel("Mẹo: Xòe 2 tay 2s để mở Menu không chạm")
        lbl_guide.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        strip_layout.addWidget(lbl_guide)

        layout.addWidget(mode_strip)
        self._update_mode_strip_ui()

        return page

    # ── 4. PAGE 1: KHO CỬ CHỈ & BẢNG THỐNG KÊ (LIBRARY & DASHBOARD) ──────────

    def _build_page_library(self) -> QWidget:
        """Trang 1: Gesture Dashboard đầy đủ — LibraryPage với stat cards, filter bar, grid và chart."""
        self.library_page = LibraryPage(db=self.db)
        return self.library_page

    def refresh_library(self) -> None:
        """Gọi LibraryPage.refresh() để load lại dữ liệu từ DB."""
        if hasattr(self, "library_page") and self.library_page is not None:
            self.library_page.refresh()

    def _create_stat_card(self, title: str, main_val: str, subtitle: str, color_hex: str) -> QFrame:
        """Tạo thẻ thống kê số liệu nổi bật."""
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;")
        lbl_v = QLabel(main_val)
        lbl_v.setStyleSheet(f"color: {color_hex}; font-size: 26px; font-weight: 800;")
        lbl_s = QLabel(subtitle)
        lbl_s.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")

        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)
        layout.addWidget(lbl_s)
        return card

    # ── 5. PAGE 2: STUDIO TRAINING & THÊM KÝ HIỆU MỚI ────────────────────────

    def _build_page_studio(self) -> QWidget:
        """Trang 2: Studio thu thập mẫu, thêm ký hiệu mới và Huấn luyện Hot-reload."""
        self.studio_page = StudioPage(
            db=self.db,
            author=self.active_author,
            camera_worker=getattr(self, "camera_worker", None),
            parent=self,
        )
        self.studio_page.gesture_added.connect(self._on_studio_gesture_added)
        self.studio_page.sample_recorded.connect(self._on_studio_sample_recorded)

        # Expose các thuộc tính và widget để tương thích ngược 100% với tests
        self.txt_gesture_name = self.studio_page.txt_gesture_name
        self.txt_display_name = self.studio_page.txt_display_name
        self.txt_guide = self.studio_page.txt_guide
        self.cbo_mode = self.studio_page.cbo_mode
        self.txt_epochs = self.studio_page.txt_epochs
        self.txt_batch = self.studio_page.txt_batch
        self.btn_train_ai = self.studio_page.btn_train_ai
        self.progress_train = self.studio_page.progress_train
        self.lbl_train_status = self.studio_page.lbl_train_status

        return self.studio_page

    # ── 6. PAGE 3: CÀI ĐẶT HỆ THỐNG (SETTINGS) ───────────────────────────────

    def _build_page_settings(self) -> QWidget:
        """Trang 3: Cấu hình độ nhạy chuột, camera, bộ lọc và lưu vào SQLite."""
        self.settings_page = SettingsPage(
            db=self.db,
            author=self.active_author,
            camera_worker=getattr(self, "camera_worker", None),
            parent=self,
        )
        self.settings_page.settings_saved.connect(self._on_settings_saved)
        self.settings_page.author_changed.connect(self._on_settings_author_changed)

        # Alias các thuộc tính để tương thích ngược hoàn toàn với test cũ
        self.txt_author = self.settings_page.txt_author
        self.slider_sens = self.settings_page.slider_sens
        self.slider_ygain = self.settings_page.slider_ygain
        self.slider_beta = self.settings_page.slider_beta
        self.slider_min_cutoff = self.settings_page.slider_min_cutoff
        self.slider_right_click = self.settings_page.slider_right_click
        self.slider_fist_grace = self.settings_page.slider_fist_grace
        self.slider_switch_mode = self.settings_page.slider_switch_mode
        self.cbo_res = self.settings_page.cbo_res
        self.cbo_camera_idx = self.settings_page.cbo_camera_idx
        self.chk_clahe = self.settings_page.chk_clahe
        self.lbl_sens_val = self.settings_page.lbl_sens_val

        return self.settings_page

    # ── LOGIC ĐIỀU KHIỂN & SỰ KIỆN ───────────────────────────────────────────

    def apply_theme(self) -> None:
        """Nạp bộ Stylesheet Dark Mode cao cấp vào ứng dụng."""
        self.setStyleSheet(MODERN_DARK_QSS)

    def _switch_page(self, index: int) -> None:
        """Chuyển đổi giữa các trang qua StackedWidget và cập nhật trạng thái nút Sidebar."""
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        titles = [
            ("TRANG CHỦ / LIVE CAMERA HUB", "Theo dõi thời gian thực & điều khiển máy tính bằng cử chỉ tay"),
            ("KHO CỬ CHỈ & THỐNG KÊ (LIBRARY)", "Danh mục cử chỉ đã đăng ký, số lượng mẫu và người đóng góp"),
            ("STUDIO THU THẬP & HUẤN LUYỆN AI", "Quay mẫu cử chỉ mới, gộp dữ liệu bạn bè và nạp mô hình Hot-Reload"),
            ("CÀI ĐẶT HỆ THỐNG", "Cấu hình độ nhạy chuột, camera và thông tin người dùng"),
        ]
        if index < len(titles):
            self.lbl_header_title.setText(titles[index][0])
            self.lbl_header_sub.setText(titles[index][1])

        if index == 1:
            self.refresh_stats()
        elif index == 2:
            if hasattr(self, "studio_page") and self.studio_page:
                self.studio_page.refresh_gestures_dropdown()

        # Tự động ngắt điều khiển chuột OS khi người dùng chuyển sang các tab quản trị / studio
        if hasattr(self, "camera_worker") and self.camera_worker:
            if index == 0:
                self.camera_worker.set_mouse_paused(self.is_paused)
            else:
                self.camera_worker.set_mouse_paused(True)

    def _on_mode_button_clicked(self, mode_key: str) -> None:
        """Xử lý khi người dùng click vào một nút trong Dải chọn Mode nhỏ."""
        self.current_mode_name = mode_key
        self.is_paused = False
        self.db.set_setting("current_mode", mode_key)
        if hasattr(self, "camera_worker") and self.camera_worker:
            self.camera_worker.set_mode(mode_key)
            self.camera_worker.set_paused(False)
        self._update_mode_strip_ui()

    def _toggle_standby(self) -> None:
        """Bật / Tắt trạng thái Tạm Khóa chuột."""
        self.is_paused = not self.is_paused
        if hasattr(self, "camera_worker") and self.camera_worker:
            self.camera_worker.set_paused(self.is_paused)
        self._update_mode_strip_ui()

    def _update_mode_strip_ui(self) -> None:
        """Cập nhật giao diện phát sáng của các nút chọn Mode bên dưới camera."""
        if self.is_paused:
            self.lbl_mode_badge.setText("⏸ ĐANG TẠM KHÓA")
            self.lbl_mode_badge.setStyleSheet(f"""
                background-color: rgba(244, 63, 94, 0.20);
                color: {COLOR_ROSE};
                border: 1px solid {COLOR_ROSE};
                border-radius: 12px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 700;
            """)
            self.btn_standby.setText("▶ MỞ KHÓA")
            for btn in self.mode_buttons.values():
                btn.setProperty("active", "false")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            if hasattr(self, "pip_widget") and self.pip_widget:
                self.pip_widget.set_mode_badge("TẠM KHÓA", COLOR_ROSE)
            return

        self.btn_standby.setText("⏸ TẠM KHÓA (STANDBY)")
        mode_titles = {
            "MODE1_MOUSE": "MODE 1: CHUỘT (0ms)",
            "MODE2_GESTURES": "MODE 2: TRÌNH CHIẾU",
            "MODE3_SIGN": "MODE 3: KÝ HIỆU NNKH",
        }
        badge_text = mode_titles.get(self.current_mode_name, "MODE 1: CHUỘT")
        self.lbl_mode_badge.setText(badge_text)
        self.lbl_mode_badge.setStyleSheet(f"""
            background-color: rgba(16, 185, 129, 0.20);
            color: {COLOR_GREEN};
            border: 1px solid {COLOR_GREEN};
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
        """)

        for m_key, btn in self.mode_buttons.items():
            is_act = (m_key == self.current_mode_name)
            btn.setProperty("active", "true" if is_act else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if hasattr(self, "pip_widget") and self.pip_widget:
            self.pip_widget.set_mode_badge(badge_text, COLOR_GREEN)

    def _show_camera_placeholder(self) -> None:
        """Hiển thị placeholder text khi chưa có frame camera."""
        self.lbl_cam_preview.clear()
        self.lbl_cam_preview.setText(
            "📷 Đang kết nối Camera...\n"
            "Kiểm tra webcam đã được cắm vào chưa.\n\n"
            "Nếu không có camera, hệ thống sẽ chạy ở chế độ Simulation."
        )
        self.lbl_cam_preview.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px; font-weight: 500;")

    def update_camera_frame(self, frame) -> None:
        """Cập nhật khung hình camera từ OpenCV lên Canvas hiển thị."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        try:
            import cv2
            from PyQt6.QtGui import QImage, QPixmap

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Dữ liệu bytes cần được copy rõ ràng để tránh giải phóng bộ nhớ sớm
            q_img = QImage(rgb_frame.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            target_w = max(100, self.camera_canvas.width() - 4)
            target_h = max(100, self.camera_canvas.height() - 4)
            scaled = pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            # Xóa text placeholder khi nhận được frame đầu tiên
            if self.lbl_cam_preview.text():
                self.lbl_cam_preview.setText("")
                self.lbl_cam_preview.setStyleSheet("background-color: transparent;")
            self.lbl_cam_preview.setPixmap(scaled)
        except Exception:
            pass

    def update_hud_metrics(
        self,
        fps: float = 60.0,
        action: str = "Di chuot (0ms)",
        pipeline: str = "Dual-Pipeline (Raw + OneEuro)",
        guard: str = "HOẠT ĐỘNG (6 lớp)",
    ) -> None:
        """Cập nhật các nhãn đo đạc thời gian thực bên cạnh camera."""
        self.lbl_hud_fps.setText(f"Tốc độ khung hình: {fps:.1f} FPS")
        self.lbl_hud_action.setText(f"Hành động: {action}")
        self.lbl_hud_pipeline.setText(f"Đường truyền: {pipeline}")
        self.lbl_hud_guard.setText(f"Bảo vệ sinh học: {guard}")

    @property
    def card_total_gestures(self):
        return getattr(self.library_page, "_card_gestures", None)

    @property
    def card_total_samples(self):
        return getattr(self.library_page, "_card_samples", None)

    @property
    def card_total_contribs(self):
        return getattr(self.library_page, "_card_contribs", None)

    @property
    def grid_layout(self):
        return getattr(self.library_page, "_grid_layout", None)

    def refresh_stats(self) -> None:
        """Cập nhật danh sách cử chỉ và số liệu thống kê từ SQLite thông qua LibraryPage."""
        if hasattr(self, "library_page") and self.library_page is not None:
            self.library_page.refresh()

    def _add_custom_gesture(self) -> None:
        """Thêm cử chỉ mới vào SQLite (Ủy quyền cho StudioPage)."""
        if hasattr(self, "studio_page") and self.studio_page is not None:
            self.studio_page.add_custom_gesture()

    def _export_my_zip(self) -> None:
        """Xuất gói dữ liệu của người dùng hiện tại (Ủy quyền cho StudioPage)."""
        if hasattr(self, "studio_page") and self.studio_page is not None:
            self.studio_page.export_my_zip()

    def _import_friend_zip(self) -> None:
        """Nhập file ZIP từ bạn bè vào hệ thống (Ủy quyền cho StudioPage)."""
        if hasattr(self, "studio_page") and self.studio_page is not None:
            self.studio_page.import_friend_zip()

    def _on_studio_gesture_added(self, gesture_name: str) -> None:
        """Cập nhật dữ liệu toàn hệ thống khi có cử chỉ mới từ Studio."""
        self.refresh_stats()

    def _on_studio_sample_recorded(self, gesture_name: str) -> None:
        """Cập nhật dữ liệu toàn hệ thống khi có mẫu mới vừa thu thập."""
        self.refresh_stats()

    def _on_worker_frame_ready(self, frame) -> None:
        """Nhận frame từ CameraWorker và cập nhật lên Canvas + PiP."""
        self.update_camera_frame(frame)
        if hasattr(self, "pip_widget") and self.pip_widget and self.pip_widget.isVisible():
            pix = self.lbl_cam_preview.pixmap()
            if pix:
                self.pip_widget.update_frame(pix)

    def _on_worker_metrics_ready(
        self,
        fps: float,
        latency_ms: float,
        action: str,
        pipeline: str,
        guard: str,
    ) -> None:
        """Nhận chỉ số từ CameraWorker và cập nhật Side HUD + Header."""
        self.update_hud_metrics(fps, action, pipeline, guard)
        self.lbl_status.setText(f"{action} | {fps:.1f} FPS")

    def _on_worker_mode_switched(self, new_mode: str) -> None:
        """Xử lý khi người dùng đổi chế độ bằng cử chỉ tay (2 tay mở)."""
        self.current_mode_name = new_mode
        self.db.set_setting("current_mode", new_mode)
        self._update_mode_strip_ui()

    def _on_worker_error(self, err_msg: str) -> None:
        """Ghi nhận lỗi từ luồng camera."""
        self.lbl_status.setText(f"Lỗi Camera: {err_msg}")

    def _toggle_pip(self) -> None:
        """Bật / Tắt chế độ Cửa sổ Mini PiP ghim góc màn hình."""
        from PyQt6.QtWidgets import QApplication
        if self.pip_widget.isVisible():
            self.pip_widget.hide()
            self.btn_pip.setText("🗗  Thu Nhỏ Mini (PiP)")
        else:
            screen = QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                x = screen_geom.right() - self.pip_widget.width() - 25
                y = screen_geom.bottom() - self.pip_widget.height() - 25
                self.pip_widget.move(x, y)
            self.pip_widget.show()
            self.btn_pip.setText("🗖  Đang Mở Mini (PiP)")

    def _restore_from_pip(self) -> None:
        """Khôi phục cửa sổ chính khi click nút phóng to trên PiP."""
        self.showNormal()
        self.activateWindow()
        self.btn_pip.setText("🗗  Thu Nhỏ Mini (PiP)")

    def _run_training(self) -> None:
        """Chạy huấn luyện mô hình học sâu qua QThread không nghẽn giao diện."""
        try:
            epochs = int(self.txt_epochs.text().strip() or "20")
            batch = int(self.txt_batch.text().strip() or "32")
        except ValueError:
            epochs = 20
            batch = 32

        self.btn_train_ai.setEnabled(False)
        self.progress_train.setValue(0)
        self.lbl_train_status.setText("Trạng thái: Đang chuẩn bị dataset phân tán...")

        self.train_worker = TrainWorker(db=self.db, epochs=epochs, batch_size=batch, parent=self)
        self.train_worker.epoch_progress.connect(self._on_train_epoch_progress)
        self.train_worker.training_completed.connect(self._on_train_completed)
        self.train_worker.training_failed.connect(self._on_train_failed)
        self.train_worker.start()

    def _on_train_epoch_progress(self, ep: int, total_epochs: int, loss: float, acc: float) -> None:
        pct = int((ep / max(1, total_epochs)) * 100)
        self.progress_train.setValue(pct)
        self.lbl_train_status.setText(f"Epoch {ep}/{total_epochs} | Loss: {loss:.4f} | Acc: {acc:.1f}%")

    def _on_train_completed(self, summary: Dict[str, Any]) -> None:
        self.btn_train_ai.setEnabled(True)
        self.progress_train.setValue(100)
        self.lbl_train_status.setText(f"Huấn luyện hoàn tất! Acc: {summary.get('final_acc', 0)}% - Đã Hot-Reload.")
        QMessageBox.information(
            self,
            "Huấn luyện AI thành công",
            f"Đã huấn luyện xong {summary.get('total_samples', 0)} mẫu qua {summary.get('total_classes', 0)} cử chỉ!\n"
            f"Độ chính xác: {summary.get('final_acc', 0)}%\n"
            f"Mô hình đã tự động Hot-Reload vào Camera."
        )

    def _on_train_failed(self, err_msg: str) -> None:
        self.btn_train_ai.setEnabled(True)
        self.lbl_train_status.setText(f"Lỗi: {err_msg}")
        QMessageBox.warning(self, "Lỗi Huấn Luyện", err_msg)

    def _save_settings(self) -> None:
        """Lưu cài đặt vào SQLite (Ủy quyền cho SettingsPage)."""
        if hasattr(self, "settings_page") and self.settings_page:
            self.settings_page.save_settings()

    def _on_settings_saved(self, s: Dict[str, Any]) -> None:
        """Cập nhật dữ liệu toàn hệ thống khi cài đặt được lưu."""
        self.active_author = s.get("active_author", self.active_author)
        self.lbl_author_display.setText(f"👤 Tác giả: {self.active_author}")
        if hasattr(self, "studio_page") and self.studio_page:
            self.studio_page.set_active_author(self.active_author)

    def _on_settings_author_changed(self, new_author: str) -> None:
        """Cập nhật tên tác giả hiển thị trên header và các tab."""
        self.active_author = new_author
        self.lbl_author_display.setText(f"👤 Tác giả: {self.active_author}")
        if hasattr(self, "studio_page") and self.studio_page:
            self.studio_page.set_active_author(self.active_author)

    def closeEvent(self, event) -> None:
        """Dọn dẹp luồng camera và cửa sổ PiP khi đóng ứng dụng."""
        if hasattr(self, "pip_widget") and self.pip_widget:
            self.pip_widget.close()
        if hasattr(self, "camera_worker") and self.camera_worker:
            self.camera_worker.stop()
        event.accept()
