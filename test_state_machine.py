"""
Test state machine logic của CameraWorker mà không cần PyQt/MediaPipe.
Chạy bằng Python thuần (venv).
"""
import sys
import time

# ── Mock minimal dependencies ──────────────────────────────────────────────
sys.modules['PyQt6'] = type(sys)('PyQt6')
sys.modules['PyQt6.QtCore'] = type(sys)('PyQt6.QtCore')

class _FakeSignal:
    def __init__(self, *args): self.calls = []
    def connect(self, fn): pass
    def emit(self, *args): self.calls.append(args)

class _FakeQThread:
    def __init__(self, parent=None): pass

class _FakeQt:
    pass

import types
qt_core = sys.modules['PyQt6.QtCore']
qt_core.QThread = _FakeQThread
qt_core.pyqtSignal = _FakeSignal

# Stub out imports workers.py uses
for mod in ['cv2', 'numpy', 'torch', 'mediapipe',
            'src.tracking.threaded_camera', 'src.preprocessing.clahe_enhancer',
            'src.tracking.kalman_filter', 'src.tracking.anatomical_guard',
            'src.modes.mode1_mouse', 'src.features.feature_extractor',
            'src.inference.voting_smoother', 'src.actions.dispatcher']:
    sys.modules[mod] = types.ModuleType(mod)

import numpy as np_real
sys.modules['numpy'] = np_real

import queue, collections

# Stub config
config_mod = types.ModuleType('config')
config_mod.SEQUENCE_LENGTH = 45
config_mod.FEATURE_DIM = 158
config_mod.MODE_MAP = {'IDLE': 0, 'MODE1_MOUSE': 1, 'MODE2_GESTURES': 2, 'MODE3_SIGN': 3}
sys.modules['config'] = config_mod

# Stub sub-modules
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.tracking'] = types.ModuleType('src.tracking')
sys.modules['src.preprocessing'] = types.ModuleType('src.preprocessing')
sys.modules['src.modes'] = types.ModuleType('src.modes')
sys.modules['src.features'] = types.ModuleType('src.features')
sys.modules['src.inference'] = types.ModuleType('src.inference')
sys.modules['src.actions'] = types.ModuleType('src.actions')

class FakeMouseCtrl:
    action_text = ""
    def count_fingers_up(self, pts): return [0]*5
    def process(self, f, h): pass
    def cancel_click_and_drag(self): pass
    def update_settings(self, **kw): pass

class FakeAD:
    def __init__(self, q): pass
    def start(self): pass
    def stop(self): pass
    def is_alive(self): return True

sys.modules['src.modes.mode1_mouse'].MouseController = FakeMouseCtrl
sys.modules['src.actions.dispatcher'].ActionDispatcher = FakeAD
sys.modules['src.tracking.threaded_camera'].ThreadedCamera = object
sys.modules['src.preprocessing.clahe_enhancer'].CLAHEEnhancer = object
sys.modules['src.tracking.kalman_filter'].HandKalmanFilter = object
sys.modules['src.tracking.anatomical_guard'].AnatomicalGuard = object
sys.modules['src.features.feature_extractor'].extract_158d_features = lambda *a, **kw: [0.0]*158
sys.modules['src.inference.voting_smoother'].VotingSmoother = object

# ── Now import only the logic we need ────────────────────────────────────────
# Extract the classes directly via exec to avoid PyQt signal issues
src = open('src/ui/workers.py', encoding='utf-8').read()

# Replace PyQt signal declarations for testing
src_test = src.replace(
    'frame_ready = pyqtSignal(np.ndarray)',        'frame_ready = _FakeSignal()'
).replace(
    'metrics_ready = pyqtSignal(float, float, str, str, str)',  'metrics_ready = _FakeSignal()'
).replace(
    'mode_switched = pyqtSignal(str)',             'mode_switched = _FakeSignal()'
).replace(
    'feature_ready = pyqtSignal(list)',            'feature_ready = _FakeSignal()'
).replace(
    'error_occurred = pyqtSignal(str)',            'error_occurred = _FakeSignal()'
).replace(
    'show_mode_menu = pyqtSignal()',               'show_mode_menu = _FakeSignal()'
).replace(
    'quit_app = pyqtSignal()',                     'quit_app = _FakeSignal()'
).replace(
    'epoch_progress = pyqtSignal(int, int, float, float)',   'epoch_progress = _FakeSignal()'
).replace(
    'training_completed = pyqtSignal(dict)',       'training_completed = _FakeSignal()'
).replace(
    'training_failed = pyqtSignal(str)',           'training_failed = _FakeSignal()'
).replace(
    'class CameraWorker(QThread):',               'class CameraWorker(_FakeQThread):'
).replace(
    'class TrainWorker(QThread):',                'class TrainWorker(_FakeQThread):'
)

globs = {
    '__name__': '__test__',
    '_FakeSignal': _FakeSignal,
    '_FakeQThread': _FakeQThread,
    'np': np_real,
    'queue': queue,
    'collections': collections,
    'time': time,
    'config': config_mod,
    'Optional': __import__('typing').Optional,
    'Dict': __import__('typing').Dict,
    'List': __import__('typing').List,
    'Any': __import__('typing').Any,
    'Path': __import__('pathlib').Path,
    'MouseController': FakeMouseCtrl,
    'ActionDispatcher': FakeAD,
    'extract_158d_features': lambda *a, **kw: [0.0]*158,
}

# Execute only (no import of cv2, mediapipe etc.)
# We'll only extract the class definition logic
exec(compile(src_test, 'workers.py', 'exec'), globs)

CameraWorker = globs['CameraWorker']

# ── Make a fake hand with N extended fingers ──────────────────────────────
def make_hand_pts(n_fingers: int) -> list:
    """
    Tạo danh sách 21 điểm giả lập bàn tay:
    - n_fingers = số ngón duỗi (0-4, không tính ngón cái)
    - Wrist tại (0.5, 0.9), ngón duỗi: TIP.y < PIP.y
    """
    pts = []
    # wrist (0) và base points
    wrist_x, wrist_y = 0.5, 0.9
    # middle_mcp (9) ở bên phải wrist → tay phải (wrist_x < mid_mcp_x)
    # Khi tay phải: ngón cái duỗi khi thumb_tip.x < thumb_ip.x
    
    # Build 21 points (idx, x, y, z)
    for i in range(21):
        pts.append((i, wrist_x, wrist_y, 0.0))
    
    # Landmark indices: TIP=8,12,16,20  PIP=6,10,14,18  MCP=5,9,13,17
    # Đặt middle_mcp.x > wrist.x (tay phải)
    pts[9] = (9, wrist_x + 0.1, wrist_y - 0.2, 0.0)  # middle_mcp
    
    # Ngón tay duỗi: TIP.y < PIP.y (TIP cao hơn PIP trên màn hình)
    tips  = [8, 12, 16, 20]
    pips  = [6, 10, 14, 18]
    for k in range(4):
        if k < n_fingers:
            # ngón duỗi: TIP.y << PIP.y
            pts[tips[k]] = (tips[k], wrist_x, 0.2, 0.0)  # TIP cao
            pts[pips[k]] = (pips[k], wrist_x, 0.5, 0.0)  # PIP thấp
        else:
            # ngón cụp: TIP.y > PIP.y
            pts[tips[k]] = (tips[k], wrist_x, 0.8, 0.0)
            pts[pips[k]] = (pips[k], wrist_x, 0.6, 0.0)
    
    # Ngón cái: không tính trong n_fingers (để đơn giản)
    # thumb_tip (4).x > thumb_ip (3).x → ngón cái cụp (tay phải)
    pts[4] = (4, wrist_x + 0.15, 0.5, 0.0)  # thumb_tip bên phải
    pts[3] = (3, wrist_x + 0.05, 0.5, 0.0)  # thumb_ip bên trái → cụp
    
    return pts

def make_open_hand():  return make_hand_pts(4)   # 4 ngón + cái = mở
def make_fisted_hand(): return make_hand_pts(0)  # 0 ngón = nắm

# ── TESTS ──────────────────────────────────────────────────────────────────
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

print("\n=== TEST 1: Khởi động ở STANDBY ===")
w = CameraWorker()
check("app_state == STANDBY", w.app_state == "STANDBY")
check("current_mode == STANDBY", w.current_mode == "STANDBY")

print("\n=== TEST 2: count_extended_fingers ===")
for n in range(5):
    pts = make_hand_pts(n)
    got = CameraWorker.count_extended_fingers(pts)
    # For n=0: only thumb, count could be 0 or 1
    check(f"  {n} ngón → got {got} (kỳ vọng >= {n})", got >= n)

print("\n=== TEST 3: is_hand_open ===")
check("4 ngón → open=True", CameraWorker.is_hand_open(make_open_hand()))
check("0 ngón → open=False", not CameraWorker.is_hand_open(make_fisted_hand()))

print("\n=== TEST 4: is_hand_fisted ===")
check("0 ngón → fisted=True", CameraWorker.is_hand_fisted(make_fisted_hand()))
check("4 ngón → fisted=False", not CameraWorker.is_hand_fisted(make_open_hand()))

print("\n=== TEST 5: State Machine Transitions ===")

# STANDBY → MENU_OPEN via BOTH_OPEN
w2 = CameraWorker()
assert w2.app_state == "STANDBY"
w2._gesture_hold_key = "BOTH_OPEN"
w2._gesture_hold_start = time.time() - 2.0  # đã giữ đủ lâu
w2._trigger_gesture("BOTH_OPEN", time.time())
check("STANDBY + BOTH_OPEN → MENU_OPEN", w2.app_state == "MENU_OPEN")
check("show_mode_menu emitted", len(w2.show_mode_menu.calls) == 1)

# MENU_OPEN + FINGERS_1 → MODE1_MOUSE
w3 = CameraWorker()
w3.app_state = "MENU_OPEN"
w3._trigger_gesture("FINGERS_1", time.time())
check("MENU + FINGERS_1 → MODE1_MOUSE", w3.app_state == "MODE1_MOUSE")
check("mode_switched emitted MODE1_MOUSE", w3.mode_switched.calls[-1] == ("MODE1_MOUSE",))

# MENU_OPEN + FINGERS_2 → MODE2_GESTURES
w4 = CameraWorker()
w4.app_state = "MENU_OPEN"
w4._trigger_gesture("FINGERS_2", time.time())
check("MENU + FINGERS_2 → MODE2_GESTURES", w4.app_state == "MODE2_GESTURES")

# MENU_OPEN + FINGERS_3 → MODE3_SIGN
w5 = CameraWorker()
w5.app_state = "MENU_OPEN"
w5._trigger_gesture("FINGERS_3", time.time())
check("MENU + FINGERS_3 → MODE3_SIGN", w5.app_state == "MODE3_SIGN")

# MENU_OPEN + FINGERS_4 → quit_app emitted
w6 = CameraWorker()
w6.app_state = "MENU_OPEN"
w6._trigger_gesture("FINGERS_4", time.time())
check("MENU + FINGERS_4 → quit_app emitted", len(w6.quit_app.calls) == 1)

# MODE1 + BOTH_FIST → MENU_OPEN
w7 = CameraWorker()
w7.app_state = "MODE1_MOUSE"
w7._trigger_gesture("BOTH_FIST", time.time())
check("MODE1 + BOTH_FIST → MENU_OPEN", w7.app_state == "MENU_OPEN")
check("show_mode_menu emitted again", len(w7.show_mode_menu.calls) == 1)

# MODE2 + ONE_FIST → STANDBY
w8 = CameraWorker()
w8.app_state = "MODE2_GESTURES"
w8._trigger_gesture("ONE_FIST", time.time())
check("MODE2 + ONE_FIST → STANDBY", w8.app_state == "STANDBY")
check("mode_switched(STANDBY) emitted", w8.mode_switched.calls[-1] == ("STANDBY",))

# STANDBY: BOTH_FIST không trigger gì
w9 = CameraWorker()
w9.app_state = "STANDBY"
w9._trigger_gesture("BOTH_FIST", time.time())
check("STANDBY + BOTH_FIST → vẫn STANDBY", w9.app_state == "STANDBY")

print(f"\n{'='*45}")
print(f"KẾT QUẢ: {PASSED} PASS / {FAILED} FAIL")
if FAILED:
    sys.exit(1)
print("TẤT CẢ TEST PASS OK!")
