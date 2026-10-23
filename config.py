"""
config.py
=========
Cấu hình tập trung cho toàn bộ hệ sinh thái HETHONGXULITAY.
Tất cả ngưỡng tham số, đường dẫn trọng số, thiết bị phần cứng
và ánh xạ cử chỉ đều được quản lý tại đây.
"""

import os
from pathlib import Path

# ── 1. ĐƯỜNG DẪN DỰ ÁN & BẢO VỆ Ổ ĐĨA C ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR  = PROJECT_ROOT / "weights"
DATASET_DIR  = PROJECT_ROOT / "dataset"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
BENCHMARK_PLOTS_DIR = BENCHMARK_DIR / "plots"

# Tạo sẵn các thư mục cần thiết
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Chuyển hướng cache sang ổ D để bảo vệ tuyệt đối ổ C
CACHE_BASE = Path(r"D:\torch_cache")
PIP_CACHE  = Path(r"D:\pip_cache")
CACHE_BASE.mkdir(parents=True, exist_ok=True)
PIP_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE)
os.environ["TORCH_HOME"]    = str(CACHE_BASE)
os.environ["HF_HOME"]       = str(r"D:\hf_cache")

# ── 2. CAMERA & TIỀN XỬ LÝ CLAHE ──────────────────────────────────────────────
CAMERA_INDEX       = 0
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720
CAMERA_FPS         = 30
CAMERA_USE_THREADING = True
CAMERA_BUFFERSIZE  = 1

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_ENABLED      = True
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_GRID_SIZE    = (8, 8)
CLAHE_AUTO_GAMMA   = True
CLAHE_ANTI_GLARE   = True
CLAHE_EDGE_SHARPEN = True

# ── 3. MEDIAPIPE BACKEND & BỘ LỌC ĐỘNG HỌC KALMAN ────────────────────────────
MAX_NUM_HANDS            = 2
MIN_DETECTION_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE  = 0.40

# Kalman Filter (Constant Velocity Kinematics: [x, y, z, vx, vy, vz])
KALMAN_PROCESS_NOISE_VAR     = 0.05    # Nhiễu mô hình gia tốc Q
KALMAN_MEASUREMENT_NOISE_VAR = 0.80    # Nhiễu quan sát camera R
KALMAN_ERROR_COV_POST        = 1.0     # Hiệp phương sai ban đầu P
KALMAN_GRACE_FRAMES          = 3       # Dead-reckoning duy trì khi mất dấu

# ── 4. TRÍCH XUẤT ĐẶC TRƯNG 158D & BỘ ĐỆM THỜI GIAN (CONTEXT GATE) ───────────
FEATURE_DIM        = 158  # 76D hand1 + 76D hand2 + 5D inter-hand + 1D Mode ID (Context Gate)
SEQUENCE_LENGTH    = 45   # Cửa sổ trượt 45 frame (~1.5s ở 30 FPS)
SEQ_LEN            = 45   # Bí danh thuận tiện (Alias)
CANONICAL_ALIGN    = True # Chuẩn hóa xoay 360° đưa về 0° Up

# Cấu hình Context Gate phân tách Mode
MAX_MODES          = 4    # Mode 0: General/Idle, 1: Mouse, 2: Gestures/Media, 3: SignLanguage
MODE_MAP           = {
    "IDLE": 0,
    "MODE1_MOUSE": 1,
    "MODE2_GESTURES": 2,
    "MODE3_SIGN": 3,
}

# ── 5. MẠNG NƠ-RON DEEP LEARNING (MULTI-SCALE CNN + BI-LSTM + ATTENTION) ──────
MODEL_INPUT_DIM    = 158
CNN_FILTERS        = [32, 32, 32]  # So kenh cho Conv1D k=3, 7, 15
CNN_KERNEL_SIZES   = [3, 7, 15]    # Bat local, mid, global temporal patterns
LSTM_HIDDEN_SIZE   = 128
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.3
NUM_CLASSES        = 10            # So luong cu chi muc tieu ban dau

# Đường dẫn file trọng số
MODEL_PT_PATH      = WEIGHTS_DIR / "gesture_bilstm_multiscale.pt"
MODEL_ONNX_PATH    = WEIGHTS_DIR / "gesture_bilstm_multiscale.onnx"
MODEL_INT8_PATH    = WEIGHTS_DIR / "gesture_bilstm_int8.onnx"

# Thiết bị huấn luyện / suy luận
DEVICE             = "cuda:0"  # Ưu tiên GPU NVIDIA RTX 3050 Laptop

# ── 6. HẬU XỬ LÝ (MAJORITY VOTING & STATE MACHINE) ────────────────────────────
VOTING_WINDOW_SIZE    = 5     # Số frame gom để vote đa số
VOTING_MIN_RATIO      = 0.60  # Tỷ lệ đồng thuận tối thiểu (>60%)
CONFIDENCE_THRESHOLD  = 0.80  # Ngưỡng xác suất Softmax tối thiểu (>80%)
ACTION_COOLDOWN_SEC   = 0.60  # Thời gian khóa đệm chống lặp cho cử chỉ động

# Danh mục nhãn cử chỉ mặc định (Gesture Label Map)
GESTURE_CLASSES = [
    "IDLE",               # 0: Không làm gì / Nghỉ
    "PINCH_CLICK",        # 1: Chụm ngón / Click chuột trái
    "FIST_LOCK",          # 2: Nắm đấm / Khóa tạm dừng
    "OPEN_PALM",          # 3: Xòe 5 ngón / Kích hoạt
    "SWIPE_LEFT",         # 4: Vuốt sang trái / Next slide, tab
    "SWIPE_RIGHT",        # 5: Vuốt sang phải / Prev slide, tab
    "SWIPE_UP",           # 6: Hất lên trên / Cuộn lên, tăng âm
    "SWIPE_DOWN",         # 7: Hất xuống dưới / Cuộn xuống, giảm âm
    "TWO_HAND_ZOOM_IN",   # 8: Hai tay mở rộng / Phóng to (Zoom In)
    "TWO_HAND_ZOOM_OUT",  # 9: Hai tay thu hẹp / Thu nhỏ (Zoom Out)
]

# Danh mục cử chỉ hợp lệ theo từng Mode (Context Gate Filtering)
MODE_GESTURE_CLASSES = {
    0: ["IDLE"],
    1: ["IDLE", "PINCH_CLICK", "FIST_LOCK"],
    2: ["IDLE", "FIST_LOCK", "OPEN_PALM", "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN", "TWO_HAND_ZOOM_IN", "TWO_HAND_ZOOM_OUT"],
    3: ["IDLE", "FIST_LOCK", "OPEN_PALM"],
}

# ── 7. ĐIỀU KHIỂN CHUỘT THỜI GIAN THỰC (MODE 1: GEOMETRIC MOUSE CONTROLLER) ───
CONTROL_REGION_MARGIN_X        = 0.25   # Cắt 25% lề trái & phải khung hình
CONTROL_REGION_MARGIN_Y        = 0.20   # Cắt 20% lề trên & dưới khung hình
MOUSE_SENSITIVITY              = 1.25   # Hệ số nhạy chuột
MOUSE_Y_DOWN_GAIN              = 1.55   # Gia tốc công thái học trục Y khi gập cổ tay xuống
MOUSE_DEADZONE                 = 0.0    # Vùng chết tĩnh (pixel màn hình)

# One Euro Filter chống rung con trỏ
MOUSE_ONE_EURO_MIN_CUTOFF      = 1.0    # Tần số cắt tối thiểu (Hz) khi tay đứng yên
MOUSE_ONE_EURO_BETA            = 0.04   # Hệ số nhạy tốc độ khi vung tay (Zero-Lag 0ms)

# Ngưỡng chạm ngón cái vào má/đầu ngón trỏ (Click trái / Kéo thả)
MOUSE_TOUCH_ENTER              = 0.28   # Chạm sát má ngón trỏ -> Click / sạc Kéo thả
MOUSE_TOUCH_RELEASE            = 0.32   # Nhấc ngón cái ra (< 0.5s) -> Click trái tức thì
MOUSE_TOUCH_LOCK               = 0.35   # Khóa vị trí con trỏ khi ngón cái áp sát
MOUSE_DRAG_HOLD_TIME           = 0.50   # Giữ ngón cái >= 0.5s để kích hoạt Kéo thả (mouseDown)
MOUSE_DRAG_DISTANCE_PX         = 45.0   # Ngưỡng pixel di chuyển kích hoạt kéo thả
MOUSE_DRAG_DISTANCE_RATIO      = 0.45   # Ngưỡng dịch chuyển theo palm_size
MOUSE_DRAG_SCREEN_THRESHOLD    = 80.0   # Ngưỡng pixel màn hình chống giật click
MOUSE_DROP_RELEASE             = 0.38   # Ngưỡng nhả ngón cái để thả chuột (mouseUp)

# Click phải (Chữ V: 2 ngón Trỏ + Giữa xòe rộng)
RIGHT_CLICK_V_GAP_MIN          = 0.35   # Tỷ lệ khoảng cách Trỏ-Giữa / palm_size >= 0.35
RIGHT_CLICK_HOLD_TIME          = 0.35   # Giữ chữ V 0.35s để xác nhận click phải
RIGHT_CLICK_COOLDOWN           = 1.0    # Thời gian hồi giữa 2 lần click phải (giây)

# Cuộn trang (2 ngón Trỏ + Giữa khép sát)
SCROLL_MODE                    = "ANCHOR" # "ANCHOR" (Joystick Cruise) hoặc "TOUCHPAD" (Vuốt)
SCROLL_FINGER_GAP_MAX          = 0.28   # Tỷ lệ khoảng cách Trỏ-Giữa / palm_size < 0.28
SCROLL_ANCHOR_DEADZONE_RATIO   = 0.04   # Bán kính vùng chết quanh tâm neo (4% palm_size)
SCROLL_ANCHOR_MAX_OFFSET_RATIO = 0.35   # Độ lệch tâm đạt tốc độ tối đa (35% palm_size)
SCROLL_ANCHOR_MAX_SPEED        = 72.0   # Tốc độ cuộn tối đa (nấc/giây)
SCROLL_ANCHOR_MIN_SPEED        = 6.0    # Tốc độ cuộn tối thiểu (nấc/giây)
SCROLL_SENSITIVITY             = 15.0   # Fallback cho TOUCHPAD mode
SCROLL_DEADZONE_RATIO          = 0.008  # Fallback cho TOUCHPAD mode
SCROLL_KINETIC_ACCEL           = True   # Fallback cho TOUCHPAD mode
SCROLL_MAX_SPEED               = 15     # Fallback cho TOUCHPAD mode

# ── 8. BỘ LỌC GIẢI PHẪU HỌC SINH HỌC (ANATOMICAL SANITY GUARD) ───────────────
ANATOMICAL_GUARD_ENABLED       = True   # Bật bộ lọc giải phẫu học sinh học
MAX_FINGER_PALM_RATIO          = 1.25   # Chiều dài ngón / kích thước lòng bàn tay <= 1.25
MAX_SEGMENT_PALM_RATIO         = 0.85   # Chiều dài 1 đốt xương / kích thước lòng bàn tay <= 0.85
MIN_PHALANX_RATIO              = 0.35   # Tỷ lệ tối thiểu giữa các đốt xương kế cận
MAX_PHALANX_RATIO              = 3.5    # Tỷ lệ tối đa giữa các đốt xương kế cận

# ── 9. MEDIA PIPE LANDMARK INDICES ────────────────────────────────────────────
WRIST            = 0
THUMB_CMC        = 1
THUMB_MCP        = 2
THUMB_IP         = 3
THUMB_TIP        = 4
INDEX_MCP        = 5
INDEX_PIP        = 6
INDEX_DIP        = 7
INDEX_TIP        = 8
MIDDLE_MCP       = 9
MIDDLE_PIP       = 10
MIDDLE_DIP       = 11
MIDDLE_TIP       = 12
RING_MCP         = 13
RING_PIP         = 14
RING_DIP         = 15
RING_TIP         = 16
PINKY_MCP        = 17
PINKY_PIP        = 18
PINKY_DIP        = 19
PINKY_TIP        = 20

