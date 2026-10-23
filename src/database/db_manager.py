"""
src/database/db_manager.py
==========================
Trình quản lý cơ sở dữ liệu SQLite (SQLite Database Manager) cho HETHONGXULITAY.
Lưu trữ Metadata cử chỉ, mẫu thu thập, lịch sử đóng góp và cài đặt ứng dụng.
"""

import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import config


DEFAULT_DB_PATH = config.DATASET_DIR / "gestures.db"


class DatabaseManager:
    """Quản lý kết nối và các truy vấn SQLite an toàn cho ứng dụng."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối SQLite mới hỗ trợ Row factory dictionary."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Khởi tạo cấu trúc các bảng và nạp dữ liệu ban đầu nếu chưa có."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Bảng danh mục cử chỉ (gestures)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    mode_id INTEGER NOT NULL,
                    description TEXT,
                    finger_guide TEXT,
                    icon TEXT,
                    is_system INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
            """)

            # 2. Bảng mẫu dữ liệu thu thập (samples)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_name TEXT NOT NULL,
                    file_path TEXT UNIQUE NOT NULL,
                    author TEXT NOT NULL DEFAULT 'User',
                    num_frames INTEGER NOT NULL DEFAULT 45,
                    feature_dim INTEGER NOT NULL DEFAULT 158,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (gesture_name) REFERENCES gestures (name) ON DELETE CASCADE
                );
            """)

            # 3. Bảng người đóng góp (contributors)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contributors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    last_active TEXT NOT NULL
                );
            """)

            # 4. Bảng cài đặt ứng dụng (app_settings)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT
                );
            """)

            # Nạp 10 cử chỉ hệ thống mặc định nếu bảng gestures rỗng
            cursor.execute("SELECT COUNT(*) FROM gestures;")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.datetime.now().isoformat()
                default_gestures = [
                    ("IDLE", "Nghỉ / Tự do", 0, "Bàn tay mở tự nhiên, không thực hiện hành động nào.", "Mở 5 ngón thả lỏng", "hand-idle", 1, now_str),
                    ("PINCH_CLICK", "Chụm ngón / Click trái", 1, "Chạm ngón cái vào má hoặc đầu ngón trỏ để click chuột trái.", "Ngón cái chạm ngón trỏ", "mouse-click", 1, now_str),
                    ("FIST_LOCK", "Nắm đấm / Tạm khóa", 1, "Nắm chặt bàn tay để khóa cứng vị trí chuột hoặc dừng nhận diện.", "Nắm 5 ngón tay lại", "lock", 1, now_str),
                    ("OPEN_PALM", "Xòe 5 ngón / Menu", 0, "Xòe rộng 5 ngón tay giữ trong 2s để mở Menu chọn chế độ.", "Xòe thẳng 5 ngón tay", "menu", 1, now_str),
                    ("SWIPE_LEFT", "Vuốt sang trái", 2, "Vung tay dứt khoát sang trái để Next slide, chuyển tab.", "Bàn tay di chuyển nhanh sang trái", "arrow-left", 1, now_str),
                    ("SWIPE_RIGHT", "Vuốt sang phải", 2, "Vung tay dứt khoát sang phải để Prev slide, lùi tab.", "Bàn tay di chuyển nhanh sang phải", "arrow-right", 1, now_str),
                    ("SWIPE_UP", "Hất lên trên", 2, "Hất bàn tay lên trên để cuộn trang hoặc tăng âm lượng.", "Bàn tay hất lên cao", "arrow-up", 1, now_str),
                    ("SWIPE_DOWN", "Hất xuống dưới", 2, "Hất bàn tay xuống dưới để cuộn trang hoặc giảm âm lượng.", "Bàn tay hạ nhanh xuống", "arrow-down", 1, now_str),
                    ("TWO_HAND_ZOOM_IN", "Phóng to (Zoom In)", 2, "Hai bàn tay mở rộng dần khoảng cách sang hai bên.", "Hai bàn tay di chuyển xa nhau", "zoom-in", 1, now_str),
                    ("TWO_HAND_ZOOM_OUT", "Thu nhỏ (Zoom Out)", 2, "Hai bàn tay thu hẹp khoảng cách lại gần nhau.", "Hai bàn tay di chuyển lại gần nhau", "zoom-out", 1, now_str),
                ]
                cursor.executemany("""
                    INSERT INTO gestures (name, display_name, mode_id, description, finger_guide, icon, is_system, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, default_gestures)

            # Nạp cài đặt mặc định nếu bảng app_settings rỗng
            cursor.execute("SELECT COUNT(*) FROM app_settings;")
            if cursor.fetchone()[0] == 0:
                default_settings = [
                    ("active_author", "Tôi", "Tên người dùng đang thu thập mẫu hiện tại"),
                    ("current_mode", "MODE1_MOUSE", "Chế độ ứng dụng ban đầu"),
                    ("mouse_sensitivity", "1.25", "Độ nhạy con trỏ chuột"),
                    ("mouse_y_down_gain", "1.55", "Hệ số gia tốc công thái học trục Y"),
                    ("one_euro_beta", "0.04", "Hệ số nhạy vận tốc bộ lọc One Euro"),
                    ("camera_width", "1280", "Độ phân giải chiều rộng camera"),
                    ("camera_height", "720", "Độ phân giải chiều cao camera"),
                    ("camera_index", "0", "Chỉ số webcam"),
                    ("dark_mode", "1", "Giao diện Dark Mode (1=Bật, 0=Tắt)"),
                ]
                cursor.executemany("""
                    INSERT INTO app_settings (key, value, description)
                    VALUES (?, ?, ?);
                """, default_settings)

            conn.commit()

    # ── API CỬ CHỈ (GESTURES) ────────────────────────────────────────────────

    def add_gesture(
        self,
        name: str,
        display_name: str,
        mode_id: int,
        description: str = "",
        finger_guide: str = "",
        icon: str = "custom-gesture",
        is_system: int = 0,
    ) -> bool:
        """Thêm một cử chỉ / ký hiệu mới vào cơ sở dữ liệu."""
        clean_name = name.strip().upper().replace(" ", "_")
        now_str = datetime.datetime.now().isoformat()
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO gestures (name, display_name, mode_id, description, finger_guide, icon, is_system, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (clean_name, display_name, mode_id, description, finger_guide, icon, is_system, now_str))
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def delete_gesture(self, name: str) -> bool:
        """Xóa cử chỉ (chỉ cho phép xóa cử chỉ tùy chỉnh, không xóa cử chỉ hệ thống)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT is_system FROM gestures WHERE name = ?;", (name,))
            row = cursor.fetchone()
            if not row or row["is_system"] == 1:
                return False  # Cử chỉ hệ thống được bảo vệ
            cursor.execute("DELETE FROM gestures WHERE name = ?;", (name,))
            conn.commit()
            return cursor.rowcount > 0

    def get_all_gestures(self) -> List[Dict[str, Any]]:
        """Lấy toàn bộ danh sách cử chỉ sắp xếp theo mode_id và id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures ORDER BY mode_id ASC, id ASC;")
            return [dict(r) for r in cursor.fetchall()]

    def get_gesture(self, name: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết một cử chỉ theo tên mã."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures WHERE name = ?;", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_gestures_by_mode(self, mode_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách cử chỉ của một Mode cụ thể."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gestures WHERE mode_id = ? ORDER BY id ASC;", (mode_id,))
            return [dict(r) for r in cursor.fetchall()]

    # ── API MẪU THU THẬP (SAMPLES) ───────────────────────────────────────────

    def add_sample(
        self,
        gesture_name: str,
        file_path: str,
        author: str = "User",
        num_frames: int = 45,
        feature_dim: int = 158,
    ) -> bool:
        """Ghi nhận một mẫu thu thập mới vào cơ sở dữ liệu."""
        now_str = datetime.datetime.now().isoformat()
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO samples (gesture_name, file_path, author, num_frames, feature_dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (gesture_name, str(file_path), author, num_frames, feature_dim, now_str))

                # Cập nhật số lượng mẫu của tác giả
                conn.execute("""
                    INSERT INTO contributors (name, sample_count, last_active)
                    VALUES (?, 1, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        sample_count = sample_count + 1,
                        last_active = excluded.last_active;
                """, (author, now_str))

                conn.commit()
            return True
        except Exception:
            return False

    def delete_sample(self, file_path: str) -> bool:
        """Xóa một mẫu theo đường dẫn file và cập nhật lại số lượng của tác giả."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT author FROM samples WHERE file_path = ?;", (str(file_path),))
                row = cursor.fetchone()
                author = row["author"] if row else None

                cursor.execute("DELETE FROM samples WHERE file_path = ?;", (str(file_path),))
                deleted = cursor.rowcount > 0

                if deleted and author:
                    cursor.execute("""
                        UPDATE contributors
                        SET sample_count = MAX(0, sample_count - 1)
                        WHERE name = ?;
                    """, (author,))
                conn.commit()
                return deleted
        except Exception:
            return False

    def get_samples_for_gesture(self, gesture_name: str) -> List[Dict[str, Any]]:
        """Lấy toàn bộ mẫu của một cử chỉ."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM samples WHERE gesture_name = ? ORDER BY id DESC;", (gesture_name,))
            return [dict(r) for r in cursor.fetchall()]

    def get_gesture_sample_counts(self) -> Dict[str, int]:
        """Thống kê số lượng mẫu thu thập cho từng cử chỉ: {gesture_name: count}."""
        counts = {g["name"]: 0 for g in self.get_all_gestures()}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT gesture_name, COUNT(*) as cnt FROM samples GROUP BY gesture_name;")
            for row in cursor.fetchall():
                counts[row["gesture_name"]] = int(row["cnt"])
        return counts

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Lấy toàn bộ danh sách mẫu có trong database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM samples ORDER BY id ASC;")
            return [dict(r) for r in cursor.fetchall()]

    # ── API TÁC GIẢ ĐÓNG GÓP (CONTRIBUTORS) ───────────────────────────────────

    def get_contributors(self) -> List[Dict[str, Any]]:
        """Lấy danh sách các tác giả đã tham gia quay mẫu."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM contributors ORDER BY sample_count DESC;")
            return [dict(r) for r in cursor.fetchall()]

    # ── API CÀI ĐẶT ỨNG DỤNG (SETTINGS) ──────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        """Đọc một giá trị cấu hình theo key."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = ?;", (key,))
            row = cursor.fetchone()
            return str(row["value"]) if row else default

    def set_setting(self, key: str, value: Any, description: Optional[str] = None) -> None:
        """Lưu hoặc cập nhật một giá trị cấu hình."""
        str_val = str(value)
        with self.get_connection() as conn:
            if description:
                conn.execute("""
                    INSERT INTO app_settings (key, value, description)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        description = excluded.description;
                """, (key, str_val, description))
            else:
                conn.execute("""
                    INSERT INTO app_settings (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value;
                """, (key, str_val))
            conn.commit()

    def get_all_settings(self) -> Dict[str, str]:
        """Lấy toàn bộ bảng cài đặt dạng dictionary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM app_settings;")
            return {row["key"]: row["value"] for row in cursor.fetchall()}
