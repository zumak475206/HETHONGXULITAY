"""
src/ui/theme.py
===============
Hệ thống giao diện Dark Mode cao cấp (Modern Web-like Dark Theme) cho PyQt6.
Bảng màu Slate/Zinc kết hợp điểm nhấn Neon Cyan & Emerald Green chuẩn công nghệ cao.
"""

# Bảng màu chuẩn thiết kế
DARK_BG_MAIN = "#0a0e17"       # Nền cửa sổ chính
DARK_BG_SIDEBAR = "#0f172a"    # Nền thanh sidebar
DARK_BG_CARD = "#141e33"       # Nền thẻ / panel
DARK_BG_SURFACE = "#1e293b"    # Nền thành phần tương tác
DARK_BORDER = "#2a374f"        # Đường viền phân cách
DARK_BORDER_FOCUS = "#06b6d4"  # Viền khi focus / active

TEXT_PRIMARY = "#f8fafc"       # Chữ chính trắng sáng
TEXT_SECONDARY = "#94a3b8"     # Chữ phụ xám xanh
TEXT_MUTED = "#64748b"         # Chữ mờ chú thích

COLOR_CYAN = "#06b6d4"         # Neon Cyan chính
COLOR_CYAN_HOVER = "#22d3ee"   # Cyan sáng khi hover
COLOR_GREEN = "#10b981"        # Neon Emerald (Thành công / Đang bật)
COLOR_AMBER = "#f59e0b"        # Cảnh báo
COLOR_ROSE = "#f43f5e"         # Đỏ / Thoát / Dừng


MODERN_DARK_QSS = f"""
/* ── TOÀN BỘ CỬA SỔ & GIAO DIỆN CHÍNH ────────────────────────────────────── */
QMainWindow, QWidget#CentralWidget {{
    background-color: {DARK_BG_MAIN};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}}

/* ── SIDEBAR THANH ĐIỀU HƯỚNG BÊN TRÁI ────────────────────────────────────── */
QWidget#SidebarContainer {{
    background-color: {DARK_BG_SIDEBAR};
    border-right: 1px solid {DARK_BORDER};
    min-width: 250px;
    max-width: 250px;
}}

QLabel#SidebarLogoTitle {{
    color: {COLOR_CYAN};
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

QLabel#SidebarSubtitle {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 500;
}}

QPushButton.SidebarNavButton {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
    margin: 3px 10px;
}}

QPushButton.SidebarNavButton:hover {{
    background-color: rgba(6, 182, 212, 0.10);
    color: {COLOR_CYAN_HOVER};
}}

QPushButton.SidebarNavButton[active="true"] {{
    background-color: rgba(6, 182, 212, 0.18);
    color: {COLOR_CYAN_HOVER};
    border-left: 3px solid {COLOR_CYAN};
}}

/* ── HEADER KHU VỰC TRÊN CÙNG ─────────────────────────────────────────────── */
QWidget#HeaderContainer {{
    background-color: {DARK_BG_MAIN};
    border-bottom: 1px solid {DARK_BORDER};
    min-height: 58px;
    max-height: 58px;
    padding: 0px 20px;
}}

QLabel#HeaderTitle {{
    color: {TEXT_PRIMARY};
    font-size: 16px;
    font-weight: 700;
}}

QLabel#HeaderSubtitle {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

/* Thẻ pill trạng thái hệ thống */
QFrame#StatusPill {{
    background-color: {DARK_BG_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 14px;
    padding: 4px 12px;
}}

QLabel#StatusDotGreen {{
    background-color: {COLOR_GREEN};
    border-radius: 4px;
    min-width: 8px;
    max-width: 8px;
    min-height: 8px;
    max-height: 8px;
}}

QLabel#StatusText {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 600;
}}

/* ── THẺ NỘI DUNG (CARDS & PANELS) ────────────────────────────────────────── */
QFrame.CardFrame {{
    background-color: {DARK_BG_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 12px;
}}

QFrame.CardFrame:hover {{
    border: 1px solid {DARK_BORDER_FOCUS};
}}

/* ── NÚT BẤM (BUTTONS) ─────────────────────────────────────────────────────── */
QPushButton.PrimaryButton {{
    background-color: {COLOR_CYAN};
    color: #041019;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 13px;
    font-weight: 700;
}}

QPushButton.PrimaryButton:hover {{
    background-color: {COLOR_CYAN_HOVER};
}}

QPushButton.PrimaryButton:pressed {{
    background-color: #0891b2;
}}

QPushButton.SecondaryButton {{
    background-color: {DARK_BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {DARK_BORDER};
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton.SecondaryButton:hover {{
    background-color: #27354f;
    border-color: {COLOR_CYAN};
    color: {COLOR_CYAN_HOVER};
}}

QPushButton.DangerButton {{
    background-color: rgba(244, 63, 94, 0.15);
    color: {COLOR_ROSE};
    border: 1px solid rgba(244, 63, 94, 0.35);
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton.DangerButton:hover {{
    background-color: {COLOR_ROSE};
    color: #ffffff;
}}

/* Dải nút chuyển Mode nhỏ bên dưới Camera */
QPushButton.ModeStripButton {{
    background-color: {DARK_BG_SURFACE};
    color: {TEXT_SECONDARY};
    border: 1px solid {DARK_BORDER};
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 12px;
    font-weight: 700;
    margin: 0px 4px;
}}

QPushButton.ModeStripButton:hover {{
    border-color: {COLOR_CYAN};
    color: {TEXT_PRIMARY};
}}

QPushButton.ModeStripButton[active="true"] {{
    background-color: rgba(16, 185, 129, 0.20);
    color: {COLOR_GREEN};
    border: 1px solid {COLOR_GREEN};
}}

/* ── Ô NHẬP LIỆU & COMBOBOX ────────────────────────────────────────────────── */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {DARK_BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {DARK_BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {COLOR_CYAN};
    background-color: #172338;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

/* ── THANH TIẾN TRÌNH (PROGRESS BAR) ──────────────────────────────────────── */
QProgressBar {{
    background-color: {DARK_BG_SURFACE};
    border: 1px solid {DARK_BORDER};
    border-radius: 6px;
    height: 12px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-size: 10px;
    font-weight: 700;
}}

QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLOR_CYAN}, stop:1 {COLOR_GREEN});
    border-radius: 5px;
}}

/* ── THANH CUỘN (SCROLLBAR) HIỆN ĐẠI ───────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {DARK_BG_MAIN};
    width: 8px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background-color: {DARK_BORDER};
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {DARK_BG_MAIN};
    height: 8px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background-color: {DARK_BORDER};
    border-radius: 4px;
    min-width: 24px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""
