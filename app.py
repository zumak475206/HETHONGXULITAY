"""
app.py
======
Điểm khởi chạy chính của Ứng dụng Desktop Quản trị HETHONGXULITAY (PyQt6).
- Tối ưu hóa luồng & tiến trình Windows (ABOVE_NORMAL, Anti-Power-Throttling).
- Kết nối Cơ sở dữ liệu SQLite Metadata (gestures.db).
- Khởi tạo giao diện Dashboard Dark Mode hiện đại (Sidebar, Live Camera Canvas, Mode Selector Strip, Training Studio, Settings).
"""

import sys
import os
import io
import argparse
import signal
from pathlib import Path

# Đảm bảo mã hóa console utf-8 an toàn trên Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Đảm bảo đường dẫn gốc của dự án nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Thiết lập cache trên ổ D bảo vệ an toàn cho ổ C
os.environ.setdefault("PIP_CACHE_DIR", "D:\\pip_cache")
os.environ.setdefault("TORCH_HOME", "D:\\torch_cache")
os.environ.setdefault("HF_HOME", "D:\\hf_cache")

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from src.utils.process_optimizer import optimize_windows_process
from src.database.db_manager import DatabaseManager
from src.ui.theme import MODERN_DARK_QSS
from src.ui.main_window import MainWindow


def parse_args():
    """Xử lý tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="HETHONGXULITAY — AI Hand Gesture Studio & Control Center"
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Chạy ở chế độ offscreen headless (dùng cho CI/Smoke Testing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Chế độ kiểm thử khởi động nhanh (tự thoát sau khi khởi tạo thành công)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Nếu có cờ --offscreen, đặt biến môi trường cho Qt
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # 1. Tối ưu hóa tiến trình Windows cho Computer Vision & Zero-Lag Mouse
    opt_status = optimize_windows_process()
    print(
        f"[SYSTEM] Tối ưu hóa tiến trình Windows: Priority={opt_status.get('priority_set')}, "
        f"EcoQoS_Off={opt_status.get('throttling_disabled')}"
    )

    # 2. Khởi tạo QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("HETHONGXULITAY")
    app.setApplicationDisplayName("HETHONGXULITAY — Hand Gesture Studio")
    app.setStyleSheet(MODERN_DARK_QSS)

    # 3. Khởi tạo Database Manager
    db = DatabaseManager()
    print(f"[DATABASE] Kết nối SQLite thành công: {db.db_path}")

    # 4. Khởi tạo MainWindow
    window = MainWindow(db=db)
    window.show()

    # Bắt Ctrl+C trên terminal mượt mà
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sig_timer = QTimer()
    sig_timer.start(500)
    sig_timer.timeout.connect(lambda: None)

    # Kết nối sự kiện đóng cửa sổ an toàn giải phóng luồng nền
    app.aboutToQuit.connect(window.close)

    # Nếu chạy ở chế độ kiểm thử (--test)
    if args.test:
        print("[TEST] Smoke test khởi động app thành công. Đóng cửa sổ và thoát...")

        def _exit_cleanly():
            window.close()
            app.quit()

        QTimer.singleShot(300, _exit_cleanly)

    exit_code = app.exec()
    window.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
