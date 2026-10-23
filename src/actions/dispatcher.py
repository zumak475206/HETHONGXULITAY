"""
src/actions/dispatcher.py
=========================
Bộ điều phối hành động hệ thống đa luồng (Non-blocking Action Dispatcher).

Bản chất kỹ thuật:
- Chạy trên một luồng (Thread) riêng biệt để cô lập hoàn toàn các lời gọi hệ điều hành
  (PyAutoGUI, phím nóng, âm lượng) khỏi luồng AI và luồng Camera.
- Sử dụng hàng đợi Thread-Safe Queue (queue.Queue) nhận các thông điệp hành động (Action Events).
- Đảm bảo tốc độ khung hình (FPS) của thị giác máy tính không bao giờ bị nghẽn do độ trễ OS.
"""

import time
import queue
import threading
from typing import Optional, Dict, Any

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Chống crash khi chuột chạm góc màn hình
    pyautogui.PAUSE = 0.0      # Triệt tiêu độ trễ 100ms mặc định của PyAutoGUI cho phản hồi tức thì
except ImportError:
    pyautogui = None


class ActionDispatcher(threading.Thread):
    """Luồng độc lập thực thi các tác vụ chuột và phím hệ thống."""

    def __init__(self, action_queue: queue.Queue):
        super().__init__(daemon=True)
        self.action_queue = action_queue
        self.running = True
        self.last_action_name = "NONE"
        self.last_action_time = 0.0

    def run(self):
        """Vòng lặp tiêu thụ sự kiện từ hàng đợi."""
        while self.running:
            try:
                # Chờ sự kiện mới với timeout 0.1s
                event = self.action_queue.get(timeout=0.1)
                if event is None or not self.running:
                    break

                action_name = event.get("action")
                payload = event.get("payload", {})
                self._dispatch(action_name, payload)

                self.last_action_name = action_name
                self.last_action_time = time.time()
                self.action_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ActionDispatcher] Lỗi thực thi: {e}")

    def _dispatch(self, action_name: str, payload: dict):
        """Thực thi lệnh cụ thể theo ánh xạ cử chỉ."""
        if not pyautogui:
            return

        try:
            if action_name == "PINCH_CLICK":
                pyautogui.click()

            elif action_name == "SWIPE_LEFT":
                # Chuyển slide tiếp theo / Tab kế
                pyautogui.press("right")

            elif action_name == "SWIPE_RIGHT":
                # Quay lại slide trước / Tab trước
                pyautogui.press("left")

            elif action_name == "SWIPE_UP":
                # Cuộn trang lên
                pyautogui.scroll(120)

            elif action_name == "SWIPE_DOWN":
                # Cuộn trang xuống
                pyautogui.scroll(-120)

            elif action_name == "TWO_HAND_ZOOM_IN":
                # Phóng to: Ctrl + Dấu cộng
                pyautogui.hotkey("ctrl", "+")

            elif action_name == "TWO_HAND_ZOOM_OUT":
                # Thu nhỏ: Ctrl + Dấu trừ
                pyautogui.hotkey("ctrl", "-")

            elif action_name == "FIST_LOCK":
                # Tạm dừng / Play media
                pyautogui.press("space")

            elif action_name == "FAILSAFE_RELEASE":
                # Nhả chuột và các phím bổ trợ nếu bàn tay bất ngờ biến mất khỏi camera
                pyautogui.mouseUp()

        except Exception as err:
            print(f"[ActionDispatcher] Lỗi khi gọi PyAutoGUI: {err}")

    def stop(self):
        """Dừng luồng điều phối một cách an toàn (chống deadlock)."""
        self.running = False
        try:
            self.action_queue.put_nowait(None)
        except queue.Full:
            try:
                # Xả 1 slot để nhét sentinel None
                self.action_queue.get_nowait()
                self.action_queue.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass
