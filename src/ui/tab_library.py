"""
src/ui/tab_library.py
=====================
Tab 2 — Kho Cử Chỉ & Bảng Thống Kê (Gesture Dashboard) cho HETHONGXULITAY.

Chứa:
  - GestureCard(QFrame)          : Thẻ hiển thị 1 cử chỉ trong lưới 3 cột
  - GestureDetailDialog(QDialog) : Dialog chi tiết khi click nút "Chi tiết"
  - LibraryPage(QWidget)         : Trang Library đầy đủ nhận db_manager làm tham số
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
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

from src.ui.theme import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_ROSE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    DARK_BG_CARD,
    DARK_BG_SURFACE,
    DARK_BORDER,
)

# ─── hằng số ─────────────────────────────────────────────────────────────────
_TARGET_SAMPLES = 150   # mục tiêu progress bar
_COLS = 3               # số cột grid


def _mode_color(mode_id: int) -> str:
    """Trả về màu hex tương ứng với mode_id."""
    return {0: COLOR_CYAN, 1: COLOR_CYAN, 2: COLOR_GREEN, 3: COLOR_AMBER}.get(mode_id, COLOR_CYAN)


def _mode_label(mode_id: int) -> str:
    return {0: "HỆ THỐNG", 1: "MODE 1", 2: "MODE 2", 3: "MODE 3"}.get(mode_id, f"MODE {mode_id}")


def _sample_color(count: int) -> str:
    if count >= 50:
        return COLOR_GREEN
    if count >= 10:
        return COLOR_AMBER
    return COLOR_ROSE


# ─── GestureCard ─────────────────────────────────────────────────────────────

class GestureCard(QFrame):
    """
    Thẻ hiển thị thông tin 1 cử chỉ trong grid.
    Emits detail_requested(gesture_name) khi click "Chi tiết".
    """

    detail_requested: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        gesture: Dict[str, Any],
        sample_count: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.gesture_name: str = gesture.get("name", "")
        self.display_name: str = gesture.get("display_name", "")
        self._mode_id: int = int(gesture.get("mode_id", 0))
        self._build(gesture, sample_count)

    def _build(self, g: Dict[str, Any], cnt: int) -> None:
        self.setProperty("class", "CardFrame")
        self.setFixedHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(5)

        # ── Header: tên + mode badge ──────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        lbl_name = QLabel(g.get("name", ""))
        lbl_name.setStyleSheet(
            f"color: {COLOR_CYAN}; font-weight: 800; font-size: 13px; background: transparent;"
        )
        top_row.addWidget(lbl_name)
        top_row.addStretch()

        mode_id = int(g.get("mode_id", 0))
        mc = _mode_color(mode_id)
        ml = _mode_label(mode_id)
        lbl_mode = QLabel(ml)
        lbl_mode.setStyleSheet(
            f"color: {mc}; background-color: rgba(0,0,0,0.35);"
            f"border: 1px solid {mc}; border-radius: 4px;"
            f"padding: 1px 6px; font-size: 10px; font-weight: 700;"
        )
        top_row.addWidget(lbl_mode)
        layout.addLayout(top_row)

        # ── Số mẫu + progress bar ─────────────────────────────────────────────
        sc = _sample_color(cnt)
        lbl_samples = QLabel(f"📦 {cnt} mẫu")
        lbl_samples.setStyleSheet(f"color: {sc}; font-size: 11px; font-weight: 700; background: transparent;")
        layout.addWidget(lbl_samples)

        bar = QProgressBar()
        bar.setMaximum(_TARGET_SAMPLES)
        bar.setValue(min(cnt, _TARGET_SAMPLES))
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        bar.setStyleSheet(
            f"QProgressBar {{ background-color: {DARK_BG_SURFACE}; border: none; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background-color: {sc}; border-radius: 3px; }}"
        )
        layout.addWidget(bar)

        # ── Mô tả ─────────────────────────────────────────────────────────────
        desc = g.get("description") or ""
        lbl_desc = QLabel(desc[:60] + "…" if len(desc) > 60 else desc or "—")
        lbl_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # ── Trạng thái ngón tay ───────────────────────────────────────────────
        guide = g.get("finger_guide") or g.get("finger_config") or ""
        lbl_guide = QLabel(f"👆 {guide[:50]}" if guide else "")
        lbl_guide.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; background: transparent;")
        layout.addWidget(lbl_guide)

        layout.addStretch()

        # ── Nút Chi tiết ─────────────────────────────────────────────────────
        btn_detail = QPushButton("Chi tiết")
        btn_detail.setProperty("class", "SecondaryButton")
        btn_detail.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_detail.setFixedHeight(26)
        btn_detail.setStyleSheet(
            f"font-size: 11px; padding: 3px 10px; border-radius: 6px;"
            f"background-color: {DARK_BG_SURFACE}; color: {COLOR_CYAN};"
            f"border: 1px solid {COLOR_CYAN};"
        )
        btn_detail.clicked.connect(lambda: self.detail_requested.emit(self.gesture_name))
        layout.addWidget(btn_detail, alignment=Qt.AlignmentFlag.AlignRight)


# ─── GestureDetailDialog ──────────────────────────────────────────────────────

class GestureDetailDialog(QDialog):
    """Dialog hiển thị chi tiết 1 cử chỉ và cho phép xóa (nếu là custom)."""

    def __init__(
        self,
        gesture_name: str,
        db,
        on_deleted: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.gesture_name = gesture_name
        self.db = db
        self.on_deleted = on_deleted
        self.setWindowTitle(f"Chi tiết cử chỉ: {gesture_name}")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)
        self.setStyleSheet(f"background-color: #0f1929; color: {TEXT_PRIMARY};")

        # ── Lấy dữ liệu ──────────────────────────────────────────────────────
        try:
            g = self.db.get_gesture(self.gesture_name) or {}
            samples: List[Dict[str, Any]] = self.db.get_samples_for_gesture(self.gesture_name)
        except Exception:
            g = {}
            samples = []

        # ── Header ────────────────────────────────────────────────────────────
        mode_id = int(g.get("mode_id", 0))
        lbl_title = QLabel(f"{self.gesture_name}  —  {_mode_label(mode_id)}")
        lbl_title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 16px; font-weight: 800;")
        layout.addWidget(lbl_title)

        lbl_disp = QLabel(g.get("display_name", ""))
        lbl_disp.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        layout.addWidget(lbl_disp)

        lbl_desc = QLabel(g.get("description") or "Không có mô tả.")
        lbl_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # ── Tổng mẫu ─────────────────────────────────────────────────────────
        total = len(samples)
        sc = _sample_color(total)
        lbl_total = QLabel(f"Tổng số mẫu: {total}")
        lbl_total.setStyleSheet(f"color: {sc}; font-size: 13px; font-weight: 700;")
        layout.addWidget(lbl_total)

        # ── Danh sách mẫu (tối đa 8 dòng) ───────────────────────────────────
        if samples:
            samples_card = QFrame()
            samples_card.setStyleSheet(
                f"background-color: {DARK_BG_CARD}; border: 1px solid {DARK_BORDER}; border-radius: 8px; padding: 6px;"
            )
            sc_layout = QVBoxLayout(samples_card)
            sc_layout.setContentsMargins(10, 8, 10, 8)
            sc_layout.setSpacing(3)
            lbl_hdr = QLabel("CÁC MẪU ĐÃ THU THẬP:")
            lbl_hdr.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700;")
            sc_layout.addWidget(lbl_hdr)
            for s in samples[:8]:
                txt = (
                    f"  • {s.get('author','?')} — "
                    f"{s.get('num_frames', 0)} frames — "
                    f"{s.get('created_at','')[:10]}"
                )
                lbl_s = QLabel(txt)
                lbl_s.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px;")
                sc_layout.addWidget(lbl_s)
            if total > 8:
                lbl_more = QLabel(f"  … và {total - 8} mẫu khác.")
                lbl_more.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
                sc_layout.addWidget(lbl_more)
            layout.addWidget(samples_card)

        # ── Đóng góp theo tác giả ─────────────────────────────────────────────
        if samples:
            author_counts: Dict[str, int] = {}
            for s in samples:
                a = s.get("author", "?")
                author_counts[a] = author_counts.get(a, 0) + 1
            contrib_lines = ["ĐÓNG GÓP:"]
            for a, c in sorted(author_counts.items(), key=lambda x: -x[1]):
                contrib_lines.append(f"  {a}: {c} mẫu")
            lbl_contrib = QLabel("\n".join(contrib_lines))
            lbl_contrib.setStyleSheet(f"color: {COLOR_GREEN}; font-size: 12px;")
            layout.addWidget(lbl_contrib)

        # ── Nút xóa (chỉ custom gesture) + đóng ─────────────────────────────
        btn_box = QHBoxLayout()
        is_system = int(g.get("is_system", 1))
        if is_system == 0:
            btn_del = QPushButton("🗑 Xóa cử chỉ này")
            btn_del.setStyleSheet(
                f"background-color: rgba(244,63,94,0.15); color: {COLOR_ROSE};"
                f"border: 1px solid {COLOR_ROSE}; border-radius: 8px; padding: 8px 14px; font-weight: 700;"
            )
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.clicked.connect(self._confirm_delete)
            btn_box.addWidget(btn_del)
        else:
            lbl_sys = QLabel("🔒 Cử chỉ hệ thống — không thể xóa")
            lbl_sys.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
            btn_box.addWidget(lbl_sys)

        btn_box.addStretch()
        btn_close = QPushButton("Đóng")
        btn_close.setStyleSheet(
            f"background-color: {DARK_BG_SURFACE}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {DARK_BORDER}; border-radius: 8px; padding: 8px 14px;"
        )
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _confirm_delete(self) -> None:
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa cử chỉ '{self.gesture_name}'?\nThao tác này không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                ok = self.db.delete_gesture(self.gesture_name)
            except Exception:
                ok = False
            if ok:
                if self.on_deleted:
                    self.on_deleted()
                self.accept()
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể xóa cử chỉ này.")


# ─── LibraryPage ─────────────────────────────────────────────────────────────

class LibraryPage(QWidget):
    """
    Trang Library (Tab 2) hoàn chỉnh.
    Nhận db_manager làm tham số; gọi refresh() để load / reload data từ DB.
    """

    def __init__(self, db, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.db = db
        self._all_cards: List[GestureCard] = []
        self._build_ui()
        self.refresh()

    # ── Xây dựng UI tĩnh ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # ── 1. Stat cards row ─────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self._card_gestures = self._make_stat_card("TỔNG SỐ CỬ CHỈ", "0", "hệ thống + tùy chỉnh", COLOR_CYAN)
        self._card_samples = self._make_stat_card("TỔNG MẪU ĐÃ LƯU", "0", "Chuỗi 45 frames", COLOR_GREEN)
        self._card_contribs = self._make_stat_card("NGƯỜI ĐÓNG GÓP", "0", "Tác giả hoạt động", COLOR_AMBER)
        stats_row.addWidget(self._card_gestures)
        stats_row.addWidget(self._card_samples)
        stats_row.addWidget(self._card_contribs)
        root.addLayout(stats_row)

        # ── 2. Filter bar ──────────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)

        self._cbo_mode = QComboBox()
        self._cbo_mode.addItems([
            "Tất cả",
            "Mode 1: Chuột",
            "Mode 2: Media",
            "Mode 3: NNKH",
        ])
        self._cbo_mode.setFixedWidth(180)
        self._cbo_mode.currentIndexChanged.connect(self._apply_filter)

        self._txt_search = QLineEdit()
        self._txt_search.setPlaceholderText("🔍 Tìm theo tên cử chỉ...")
        self._txt_search.setFixedWidth(240)
        self._txt_search.textChanged.connect(self._apply_filter)

        btn_refresh = QPushButton("🔄 Làm mới")
        btn_refresh.setStyleSheet(
            f"background-color: {DARK_BG_SURFACE}; color: {COLOR_CYAN};"
            f"border: 1px solid {COLOR_CYAN}; border-radius: 8px; padding: 7px 14px; font-weight: 700;"
        )
        btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_refresh.clicked.connect(self.refresh)

        lbl_filter = QLabel("BỘ LỌC:")
        lbl_filter.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700;")

        filter_row.addWidget(lbl_filter)
        filter_row.addWidget(self._cbo_mode)
        filter_row.addWidget(self._txt_search)
        filter_row.addWidget(btn_refresh)
        filter_row.addStretch()
        root.addLayout(filter_row)

        # ── 3. Gesture Grid trong ScrollArea ──────────────────────────────────
        lbl_grid_title = QLabel("DANH MỤC CỬ CHỈ ĐÃ ĐĂNG KÝ")
        lbl_grid_title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700; letter-spacing: 0.3px;"
        )
        root.addWidget(lbl_grid_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background: transparent;")
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(12)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._grid_container)
        root.addWidget(scroll, stretch=1)

        # ── 4. Contributor ASCII chart ────────────────────────────────────────
        self._chart_card = QFrame()
        self._chart_card.setProperty("class", "CardFrame")
        self._chart_card.setStyleSheet(
            f"background-color: {DARK_BG_CARD}; border: 1px solid #2a374f; border-radius: 10px;"
        )
        chart_layout = QVBoxLayout(self._chart_card)
        chart_layout.setContentsMargins(14, 12, 14, 12)
        chart_layout.setSpacing(6)

        lbl_chart_title = QLabel("THỐNG KÊ ĐÓNG GÓP")
        lbl_chart_title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;")
        chart_layout.addWidget(lbl_chart_title)

        self._lbl_chart = QLabel("—")
        self._lbl_chart.setStyleSheet(
            f"color: {COLOR_GREEN}; font-family: 'Consolas', monospace; font-size: 12px;"
        )
        self._lbl_chart.setWordWrap(False)
        chart_layout.addWidget(self._lbl_chart)

        self._chart_card.setMaximumHeight(160)
        root.addWidget(self._chart_card)

    # ── refresh() — entry point chính ─────────────────────────────────────────

    def refresh(self) -> None:
        """Load lại dữ liệu từ DB và cập nhật toàn bộ UI."""
        try:
            gestures: List[Dict[str, Any]] = self.db.get_all_gestures()
        except Exception:
            gestures = []

        try:
            counts: Dict[str, int] = self.db.get_gesture_sample_counts()
        except Exception:
            counts = {}

        try:
            contribs: List[Dict[str, Any]] = self.db.get_contributors()
        except Exception:
            contribs = []

        total_samples = sum(counts.values())

        # Cập nhật stat cards
        self._set_stat_value(self._card_gestures, str(len(gestures)))
        self._set_stat_value(self._card_samples, str(total_samples))
        self._set_stat_value(self._card_contribs, str(len(contribs)))

        # Dọn grid cũ
        self._clear_grid()
        self._all_cards = []

        # Dựng lại gesture cards
        for idx, g in enumerate(gestures):
            cnt = counts.get(g.get("name", ""), 0)
            card = GestureCard(g, cnt)
            card.detail_requested.connect(self._open_detail)
            self._all_cards.append(card)
            r, c = divmod(idx, _COLS)
            self._grid_layout.addWidget(card, r, c)

        # Cập nhật chart
        self._render_contributor_chart(contribs)

        # Áp dụng filter hiện tại (không reset combo/search)
        self._apply_filter()

    def _open_detail(self, gesture_name: str) -> None:
        """Mở GestureDetailDialog cho gesture_name."""
        dlg = GestureDetailDialog(
            gesture_name=gesture_name,
            db=self.db,
            on_deleted=self.refresh,
            parent=self,
        )
        dlg.exec()

    # ── Filter ────────────────────────────────────────────────────────────────

    def _apply_filter(self) -> None:
        """Ẩn/hiện GestureCard dựa trên mode filter và search text."""
        mode_idx = self._cbo_mode.currentIndex()   # 0=all, 1=mode1, 2=mode2, 3=mode3
        search = self._txt_search.text().strip().lower()

        for card in self._all_cards:
            name_lower = card.gesture_name.lower()
            # mode filter
            if mode_idx > 0:
                # tìm mode_id từ gesture name bằng cách tra thẳng property lưu sẵn
                # GestureCard lưu gesture_name; cần biết mode_id của nó
                # Dùng objectName trick: lưu mode_id vào card
                card_mode = getattr(card, "_mode_id", -1)
                if card_mode != mode_idx:
                    card.setVisible(False)
                    continue
            # search filter
            disp_lower = getattr(card, "display_name", "").lower()
            if search and (search not in name_lower and search not in disp_lower):
                card.setVisible(False)
                continue
            card.setVisible(True)

    # ── Contributor ASCII chart ────────────────────────────────────────────────

    def _render_contributor_chart(self, contribs: List[Dict[str, Any]]) -> None:
        if not contribs:
            self._lbl_chart.setText("(Chưa có dữ liệu đóng góp)")
            return

        max_count = max((c.get("sample_count", 0) for c in contribs), default=1) or 1
        bar_max_len = 20  # độ dài thanh tối đa
        lines = []
        for c in contribs[:6]:
            name = (c.get("name") or "?")[:8].ljust(8)
            cnt = int(c.get("sample_count", 0))
            bar_len = max(1, round(cnt / max_count * bar_max_len))
            bar = "█" * bar_len
            lines.append(f"{name}  {bar:<{bar_max_len}}  {cnt} mẫu")

        self._lbl_chart.setText("\n".join(lines))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _clear_grid(self) -> None:
        """Xóa toàn bộ widget trong grid layout."""
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    @staticmethod
    def _make_stat_card(title: str, val: str, subtitle: str, color: str) -> QFrame:
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;")
        lbl_v = QLabel(val)
        lbl_v.setObjectName("_stat_value")
        lbl_v.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: 800;")
        lbl_s = QLabel(subtitle)
        lbl_s.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")

        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)
        layout.addWidget(lbl_s)
        return card

    @staticmethod
    def _set_stat_value(card: QFrame, value: str) -> None:
        """Tìm QLabel stat value trong card và set text."""
        labels = card.findChildren(QLabel)
        if len(labels) >= 2:
            labels[1].setText(value)

