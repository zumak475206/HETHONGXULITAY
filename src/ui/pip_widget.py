"""
src/ui/pip_widget.py
====================
Cửa sổ Mini Floating Camera (Picture-in-Picture - PiP) ghim góc màn hình.
- Luôn hiển thị trên cùng (Always on Top), không viền (Frameless), bo góc tròn cao cấp.
- Cho phép người dùng vừa lướt web, làm việc, vừa theo dõi khung camera và tư thế tay.
- Hỗ trợ kéo rê di chuyển vị trí linh hoạt (Draggable) trên toàn màn hình.
"""

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)

from src.ui.theme import COLOR_CYAN, COLOR_GREEN, TEXT_PRIMARY, TEXT_MUTED


class PiPWidget(QWidget):
    """Cửa sổ nổi Mini Floating Camera (PiP)."""

    restore_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Cửa sổ không viền, luôn nổi trên cùng
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.setFixedSize(360, 230)
        self._drag_pos = QPoint()

        self._init_ui()

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # Khung chứa chính viền Neon
        self.container = QFrame(self)
        self.container.setObjectName("PiPContainer")
        self.container.setStyleSheet("""
            QFrame#PiPContainer {
                background-color: #0b1120;
                border: 2px solid #06b6d4;
                border-radius: 12px;
            }
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(8, 6, 8, 8)
        container_layout.setSpacing(4)

        # Thanh tiêu đề mini trên cùng
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 0, 2, 0)

        self.lbl_badge = QLabel("MODE 1: CHUỘT (0ms)")
        self.lbl_badge.setStyleSheet(f"""
            color: {COLOR_GREEN};
            font-size: 10px;
            font-weight: 700;
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 6px;
            padding: 2px 6px;
        """)
        top_bar.addWidget(self.lbl_badge)
        top_bar.addStretch()

        # Nút phóng to khôi phục cửa sổ chính
        self.btn_restore = QPushButton("↗ Phóng to")
        self.btn_restore.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_restore.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #06b6d4;
                color: #050811;
            }
        """)
        self.btn_restore.clicked.connect(self._on_restore)
        top_bar.addWidget(self.btn_restore)

        # Nút đóng PiP
        self.btn_close = QPushButton("✕")
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                font-size: 12px;
                font-weight: 700;
                padding: 2px 4px;
            }
            QPushButton:hover {
                color: #f43f5e;
            }
        """)
        self.btn_close.clicked.connect(self.hide)
        top_bar.addWidget(self.btn_close)

        container_layout.addLayout(top_bar)

        # Màn hình hiển thị camera
        self.lbl_cam = QLabel()
        self.lbl_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cam.setStyleSheet("""
            background-color: #050811;
            border-radius: 8px;
        """)
        container_layout.addWidget(self.lbl_cam, stretch=1)

        root_layout.addWidget(self.container)

    def update_frame(self, pixmap: QPixmap) -> None:
        """Cập nhật frame hiển thị trên PiP."""
        if pixmap and not pixmap.isNull():
            w = self.lbl_cam.width()
            h = self.lbl_cam.height()
            if w > 10 and h > 10:
                scaled = pixmap.scaled(
                    w,
                    h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.lbl_cam.setPixmap(scaled)

    def set_mode_badge(self, text: str, color_hex: str = COLOR_GREEN) -> None:
        """Cập nhật huy hiệu chế độ."""
        self.lbl_badge.setText(text)
        self.lbl_badge.setStyleSheet(f"""
            color: {color_hex};
            font-size: 10px;
            font-weight: 700;
            background-color: rgba(6, 182, 212, 0.15);
            border: 1px solid {color_hex};
            border-radius: 6px;
            padding: 2px 6px;
        """)

    def _on_restore(self) -> None:
        self.hide()
        self.restore_requested.emit()

    # Kéo thả di chuyển cửa sổ (Window Dragging)
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton and not self._drag_pos.isNull():
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
