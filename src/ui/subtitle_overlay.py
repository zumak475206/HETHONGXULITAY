"""
src/ui/subtitle_overlay.py
==========================
Cửa sổ Phụ Đề Nổi Toàn Màn Hình Máy Tính (Global Desktop Overlay Subtitle Bar).

Đặc điểm kỹ thuật:
- Hoạt động toàn hệ thống Windows (Global Always-On-Top Overlay):
  Hiển thị nổi trên mọi phần mềm (PowerPoint toàn màn hình, Zoom, Word, Chrome, Desktop...).
- Không cướp tiêu điểm chuột / bàn phím (WindowDoesNotAcceptFocus).
- Nền Acrylic Glass trong suốt bo góc tròn, viền dạ quang Neon Cyan / Emerald.
- Hỗ trợ tiếng Việt có dấu Unicode to rõ, phong cách Cinematic Subtitles.
- Tự động hiện khi ở Mode 3 (Ngôn ngữ ký hiệu), tự động ẩn khi chuyển Mode khác.
- Cho phép người dùng kéo thả chuột để di chuyển vị trí thanh phụ đề tùy ý trên màn hình.
"""

import time
from typing import Optional, Dict, Any

from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QGuiApplication
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QApplication,
    QGraphicsDropShadowEffect,
)

from src.ui.theme import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_AMBER,
    COLOR_ROSE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)


class DesktopSubtitleOverlay(QWidget):
    """
    Thanh Phụ Đề Nổi Toàn Màn Hình (Global Desktop Subtitle Overlay).
    """

    clear_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Cài đặt cờ cửa sổ nổi toàn hệ thống
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._drag_pos = QPoint()
        self._is_dragging = False
        self._active_mode = "STANDBY"
        self._last_update_time = time.time()
        self._auto_hide_sec = 4.5

        # Timer tự động ẩn khi không có phụ đề mới
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(500)
        self._fade_timer.timeout.connect(self._check_auto_hide)
        self._fade_timer.start()

        self._init_ui()
        self._reposition_bottom_center()

    def _init_ui(self) -> None:
        """Xây dựng giao diện thanh phụ đề nổi sang trọng."""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        # Container card chính
        self.card = QFrame()
        self.card.setObjectName("SubtitleCard")
        self.card.setStyleSheet("""
            QFrame#SubtitleCard {
                background-color: rgba(10, 16, 32, 0.90);
                border: 1.5px solid rgba(0, 240, 255, 0.65);
                border-radius: 18px;
            }
        """)

        # Hiệu ứng bóng đổ phát sáng Neon
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 220, 255, 90))
        shadow.setOffset(0, 4)
        self.card.setGraphicsEffect(shadow)

        card_layout = QHBoxLayout(self.card)
        card_layout.setContentsMargins(20, 12, 18, 12)
        card_layout.setSpacing(14)

        # 1. Icon & Badge tiêu đề
        badge_box = QVBoxLayout()
        badge_box.setSpacing(2)
        badge_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_badge = QLabel("🤟 PHỤ ĐỀ VSL")
        self.lbl_badge.setStyleSheet(f"""
            color: {COLOR_CYAN};
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            background: rgba(0, 240, 255, 0.15);
            padding: 3px 8px;
            border-radius: 6px;
        """)
        self.lbl_confidence = QLabel("100%")
        self.lbl_confidence.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 10px; font-weight: 700;")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)

        badge_box.addWidget(self.lbl_badge)
        badge_box.addWidget(self.lbl_confidence)
        card_layout.addLayout(badge_box)

        # Đường kẻ phân cách
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 0.15); max-width: 1px; margin: 4px 0;")
        card_layout.addWidget(sep)

        # 2. Nội dung phụ đề chính
        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        self.lbl_current_word = QLabel("Đang chờ ký hiệu...")
        self.lbl_current_word.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")

        self.lbl_sentence = QLabel("Sẵn sàng dịch ngôn ngữ ký hiệu")
        self.lbl_sentence.setStyleSheet(f"""
            color: #ffffff;
            font-size: 19px;
            font-weight: 700;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        """)
        self.lbl_sentence.setWordWrap(True)

        text_box.addWidget(self.lbl_current_word)
        text_box.addWidget(self.lbl_sentence)
        card_layout.addLayout(text_box, stretch=1)

        # 3. Nút tiện ích: Xóa câu & Sao chép
        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)

        self.btn_copy = QPushButton("📋")
        self.btn_copy.setToolTip("Sao chép phụ đề vào Clipboard")
        self.btn_copy.setFixedSize(32, 32)
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.10);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.20);
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: rgba(0, 240, 255, 0.25);
                border-color: #00f0ff;
            }
        """)
        self.btn_copy.clicked.connect(self._copy_sentence)

        self.btn_clear = QPushButton("✕")
        self.btn_clear.setToolTip("Xóa câu phụ đề hiện tại")
        self.btn_clear.setFixedSize(32, 32)
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background: rgba(244, 63, 94, 0.15);
                color: #f43f5e;
                border: 1px solid rgba(244, 63, 94, 0.35);
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(244, 63, 94, 0.30);
                border-color: #f43f5e;
            }
        """)
        self.btn_clear.clicked.connect(self._clear_sentence)

        btn_box.addWidget(self.btn_copy)
        btn_box.addWidget(self.btn_clear)
        card_layout.addLayout(btn_box)

        root.addWidget(self.card)
        self.setMinimumWidth(560)
        self.setMaximumWidth(880)

    def _reposition_bottom_center(self) -> None:
        """Căn giữa đáy màn hình máy tính."""
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            w = 640
            h = 85
            x = geom.x() + (geom.width() - w) // 2
            y = geom.y() + geom.height() - h - 50  # Cách đáy màn hình 50px
            self.setGeometry(x, y, w, h)

    def set_active_mode(self, mode_name: str) -> None:
        """Chỉ hiển thị phụ đề nổi khi ở Mode 3."""
        self._active_mode = mode_name
        if mode_name == "MODE3_SIGN":
            self.show()
            self._last_update_time = time.time()
        else:
            self.hide()

    def update_subtitle_data(
        self,
        sentence: str,
        current_word: str = "",
        current_meaning: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Cập nhật nội dung phụ đề từ luồng nhận diện."""
        now = time.time()

        if sentence or current_meaning:
            self._last_update_time = now
            if self._active_mode == "MODE3_SIGN" and not self.isVisible():
                self.show()

            # Hiển thị câu đầy đủ hoặc ý nghĩa hiện tại
            display_text = sentence if sentence else current_meaning
            self.lbl_sentence.setText(display_text)

            if current_meaning:
                self.lbl_current_word.setText(f"Đang phát hiện: {current_meaning} ({current_word})")
            else:
                self.lbl_current_word.setText("Đang theo dõi cử chỉ...")

            if confidence > 0.0:
                self.lbl_confidence.setText(f"{int(confidence * 100)}%")
                self.lbl_confidence.show()
            else:
                self.lbl_confidence.hide()

    def _check_auto_hide(self) -> None:
        """Tự động làm mờ hoặc ẩn phụ đề nếu người dùng ngừng thao tác quá 4.5s."""
        if not self.isVisible() or self._active_mode != "MODE3_SIGN":
            return
        now = time.time()
        if now - self._last_update_time > self._auto_hide_sec:
            # Thu gọn text
            if self.lbl_sentence.text() != "Sẵn sàng dịch ngôn ngữ ký hiệu":
                self.lbl_sentence.setText("Sẵn sàng dịch ngôn ngữ ký hiệu")
                self.lbl_current_word.setText("Đang chờ ký hiệu...")
                self.lbl_confidence.hide()

    def _copy_sentence(self) -> None:
        """Sao chép câu phụ đề vào khay nhớ tạm (Clipboard)."""
        text = self.lbl_sentence.text().strip()
        if text and text != "Sẵn sàng dịch ngôn ngữ ký hiệu":
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)
                self.lbl_current_word.setText("✓ Đã sao chép vào khay nhớ tạm!")

    def _clear_sentence(self) -> None:
        """Xóa câu phụ đề."""
        self.lbl_sentence.setText("Sẵn sàng dịch ngôn ngữ ký hiệu")
        self.lbl_current_word.setText("Đang chờ ký hiệu...")
        self.lbl_confidence.hide()
        self.clear_requested.emit()

    # ── Kéo thả cửa sổ tự do trên màn hình ───────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        event.accept()
