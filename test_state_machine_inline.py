"""
Inline state machine test - không cần spawn process ngoài.
Tất cả logic được copy/paste trực tiếp và test tại đây.
"""
import sys, time, queue, collections
from typing import Optional, Dict, List, Any
from pathlib import Path

PASSED = 0
FAILED = 0

def check(name, cond, extra=""):
    global PASSED, FAILED
    if cond:
        print(f"  [PASS] {name}")
        PASSED += 1
    else:
        print(f"  [FAIL] {name} {extra}")
        FAILED += 1

# ── Copy logic từ workers.py ──────────────────────────────────────────────

def count_extended_fingers(hand_pts: list) -> int:
    if not hand_pts or len(hand_pts) < 21:
        return 0
    lm: Dict[int, tuple] = {p[0]: (p[1], p[2], p[3]) for p in hand_pts}
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    count = sum(1 for tip_i, pip_i in finger_pairs if lm[tip_i][1] < lm[pip_i][1])
    wrist_x = lm[0][0]
    mid_mcp_x = lm[9][0]
    if wrist_x < mid_mcp_x:
        if lm[4][0] < lm[3][0]:
            count += 1
    else:
        if lm[4][0] > lm[3][0]:
            count += 1
    return min(5, count)

def is_hand_open(hand_pts):  return count_extended_fingers(hand_pts) >= 4
def is_hand_fisted(hand_pts): return count_extended_fingers(hand_pts) <= 1

# ── Fake CameraWorker state machine (simplified) ─────────────────────────

class FakeSignal:
    def __init__(self): self.calls = []
    def emit(self, *args): self.calls.append(args)

class FakeWorker:
    GESTURE_HOLD_SEC = 1.5
    def __init__(self):
        self.app_state = "STANDBY"
        self.current_mode = "STANDBY"
        self.is_paused = False
        self._gesture_hold_key = ""
        self._gesture_hold_start = 0.0
        self._gesture_cooldown = 0.0
        self.show_mode_menu = FakeSignal()
        self.quit_app = FakeSignal()
        self.mode_switched = FakeSignal()
        # fake temporal_buffer
        self.temporal_buffer = collections.deque(maxlen=45)

    def _trigger_gesture(self, gesture_key, now):
        self._gesture_hold_key = ""
        self._gesture_hold_start = 0.0
        self._gesture_cooldown = now + 2.0

        if gesture_key == "BOTH_OPEN" and self.app_state == "STANDBY":
            self.app_state = "MENU_OPEN"
            self.current_mode = "STANDBY"
            self.show_mode_menu.emit()

        elif gesture_key == "FINGERS_1" and self.app_state == "MENU_OPEN":
            self._enter_mode("MODE1_MOUSE")
        elif gesture_key == "FINGERS_2" and self.app_state == "MENU_OPEN":
            self._enter_mode("MODE2_GESTURES")
        elif gesture_key == "FINGERS_3" and self.app_state == "MENU_OPEN":
            self._enter_mode("MODE3_SIGN")
        elif gesture_key == "FINGERS_4" and self.app_state == "MENU_OPEN":
            self.quit_app.emit()

        elif gesture_key == "BOTH_FIST" and self.app_state not in ("STANDBY", "MENU_OPEN"):
            self.app_state = "MENU_OPEN"
            self.current_mode = "STANDBY"
            self.show_mode_menu.emit()

        elif gesture_key == "ONE_FIST" and self.app_state not in ("STANDBY", "MENU_OPEN"):
            self.app_state = "STANDBY"
            self.current_mode = "STANDBY"
            self.is_paused = False
            self.mode_switched.emit("STANDBY")

    def _enter_mode(self, mode_name):
        self.app_state = mode_name
        self.current_mode = mode_name
        self.is_paused = False
        self.temporal_buffer.clear()
        self.mode_switched.emit(mode_name)

# ── Fake hand data ─────────────────────────────────────────────────────────

def make_hand(n_fingers):
    pts = [(i, 0.5, 0.9, 0.0) for i in range(21)]
    # middle_mcp > wrist.x (tay phải)
    pts[9] = (9, 0.65, 0.6, 0.0)
    tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
    for k in range(4):
        if k < n_fingers:
            pts[tips[k]] = (tips[k], 0.5, 0.2, 0.0)  # TIP cao (duỗi)
            pts[pips[k]] = (pips[k], 0.5, 0.5, 0.0)
        else:
            pts[tips[k]] = (tips[k], 0.5, 0.8, 0.0)  # TIP thấp (cụp)
            pts[pips[k]] = (pips[k], 0.5, 0.6, 0.0)
    # Ngón cái cụp (tay phải: thumb_tip.x > thumb_ip.x)
    pts[4] = (4, 0.65, 0.5, 0.0)
    pts[3] = (3, 0.55, 0.5, 0.0)
    return pts

open_hand = make_hand(4)
fist_hand = make_hand(0)

# ── Run Tests ───────────────────────────────────────────────────────────────

print("\n=== TEST 1: Finger counting ===")
for n in range(5):
    got = count_extended_fingers(make_hand(n))
    check(f"make_hand({n}) → count={got} (expect>={n})", got >= n, f"got={got}")

print("\n=== TEST 2: Hand shape detection ===")
check("open_hand is_hand_open", is_hand_open(open_hand))
check("fist_hand NOT is_hand_open", not is_hand_open(fist_hand))
check("fist_hand is_hand_fisted", is_hand_fisted(fist_hand))
check("open_hand NOT is_hand_fisted", not is_hand_fisted(open_hand))

print("\n=== TEST 3: State transitions ===")
now = time.time()

# STANDBY → MENU_OPEN
w = FakeWorker()
assert w.app_state == "STANDBY"
w._trigger_gesture("BOTH_OPEN", now)
check("STANDBY + BOTH_OPEN → MENU_OPEN", w.app_state == "MENU_OPEN")
check("show_mode_menu.emit() called", len(w.show_mode_menu.calls) == 1)

# MENU + FINGERS_1 → MODE1_MOUSE
w = FakeWorker(); w.app_state = "MENU_OPEN"
w._trigger_gesture("FINGERS_1", now)
check("MENU + FINGERS_1 → MODE1_MOUSE", w.app_state == "MODE1_MOUSE")
check("mode_switched('MODE1_MOUSE')", w.mode_switched.calls == [("MODE1_MOUSE",)])

# MENU + FINGERS_2 → MODE2_GESTURES
w = FakeWorker(); w.app_state = "MENU_OPEN"
w._trigger_gesture("FINGERS_2", now)
check("MENU + FINGERS_2 → MODE2_GESTURES", w.app_state == "MODE2_GESTURES")

# MENU + FINGERS_3 → MODE3_SIGN
w = FakeWorker(); w.app_state = "MENU_OPEN"
w._trigger_gesture("FINGERS_3", now)
check("MENU + FINGERS_3 → MODE3_SIGN", w.app_state == "MODE3_SIGN")

# MENU + FINGERS_4 → quit_app
w = FakeWorker(); w.app_state = "MENU_OPEN"
w._trigger_gesture("FINGERS_4", now)
check("MENU + FINGERS_4 → quit_app.emit()", len(w.quit_app.calls) == 1)
check("app_state không đổi (quit xử lý bởi UI)", w.app_state == "MENU_OPEN")

# MODE1 + BOTH_FIST → MENU_OPEN
w = FakeWorker(); w.app_state = "MODE1_MOUSE"
w._trigger_gesture("BOTH_FIST", now)
check("MODE1 + BOTH_FIST → MENU_OPEN", w.app_state == "MENU_OPEN")
check("show_mode_menu called", len(w.show_mode_menu.calls) == 1)

# MODE2 + ONE_FIST → STANDBY
w = FakeWorker(); w.app_state = "MODE2_GESTURES"
w._trigger_gesture("ONE_FIST", now)
check("MODE2 + ONE_FIST → STANDBY", w.app_state == "STANDBY")
check("mode_switched('STANDBY') emitted", w.mode_switched.calls == [("STANDBY",)])

# MODE3 + ONE_FIST → STANDBY
w = FakeWorker(); w.app_state = "MODE3_SIGN"
w._trigger_gesture("ONE_FIST", now)
check("MODE3 + ONE_FIST → STANDBY", w.app_state == "STANDBY")

# Không trigger khi sai state
w = FakeWorker(); w.app_state = "STANDBY"
w._trigger_gesture("BOTH_FIST", now)
check("STANDBY + BOTH_FIST → giữ STANDBY", w.app_state == "STANDBY")
check("show_mode_menu KHÔNG emit", len(w.show_mode_menu.calls) == 0)

w = FakeWorker(); w.app_state = "STANDBY"
w._trigger_gesture("FINGERS_1", now)
check("STANDBY + FINGERS_1 → giữ STANDBY", w.app_state == "STANDBY")

print("\n=== TEST 4: Cooldown ===")
w = FakeWorker(); w.app_state = "STANDBY"
w._trigger_gesture("BOTH_OPEN", now)
cooldown_set = w._gesture_cooldown > now
check("Cooldown được set sau trigger", cooldown_set, f"cooldown={w._gesture_cooldown:.2f} now={now:.2f}")

print(f"\n{'='*45}")
print(f"KET QUA: {PASSED} PASS / {FAILED} FAIL")
if FAILED == 0:
    print("ALL TESTS PASSED!")
    sys.exit(0)
else:
    sys.exit(1)
