"""
main.py
=======
Ứng dụng chính của Hệ thống Nhận diện Cử chỉ Bàn tay (HETHONGXULITAY).

Kiến trúc Đa luồng 3 Tầng (3-Thread Pipeline):
- Luồng 1 (Camera Grabber): Đọc webcam ở tốc độ tối đa, loại bỏ frame cũ để chống trễ (Zero-Lag).
- Luồng 2 (AI Worker): CLAHE -> MediaPipe -> Kalman Filter -> 157D Features -> 30-frame Circular Buffer
                        -> Multi-Scale Bi-LSTM + Attention Inference -> 5-Frame Majority Voting.
- Luồng 3 (Action Dispatcher): Nhận sự kiện cử chỉ và thực thi lệnh hệ điều hành ngầm.

Giao diện HUD:
- Khung xương bàn tay đã làm mượt bằng Kalman.
- Thanh đo chú ý thời gian thực (Temporal Attention Heatmap Bar).
- Nhãn cử chỉ ổn định, độ tin cậy và FPS thời gian thực.
"""

import os
import sys
import time
import queue
import threading
import collections
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import torch

# Đảm bảo đường dẫn import
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing.clahe_enhancer import CLAHEEnhancer
from src.tracking.kalman_filter import HandKalmanFilter
from src.tracking.anatomical_guard import AnatomicalGuard
from src.modes.mode1_mouse import MouseController
from src.features.feature_extractor import extract_158d_features, extract_157d_features, extract_112d_features
from src.models.bilstm_attention import MultiScaleBiLSTMAttention
from src.inference.voting_smoother import VotingSmoother
from src.actions.dispatcher import ActionDispatcher
from src.tracking.threaded_camera import ThreadedCamera
from src.utils.process_optimizer import optimize_windows_process

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    mp_hands = None
    mp_drawing = None


def draw_attention_heatmap(frame: np.ndarray, attention_weights: np.ndarray, x: int = 30, y: int = 420, w: int = 240, h: int = 24):
    """Vẽ thanh biểu đồ nhiệt Attention Heatmap hiển thị trọng số của 30 frame."""
    if attention_weights is None or len(attention_weights) == 0:
        return

    weights = np.array(attention_weights, dtype=np.float32)
    # Chuẩn hóa min-max để hiển thị rõ
    w_min, w_max = float(np.min(weights)), float(np.max(weights))
    norm_w = (weights - w_min) / max(1e-5, (w_max - w_min))

    cv2.rectangle(frame, (x - 2, y - 22), (x + w + 2, y + h + 2), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, f"ATTENTION HEATMAP ({len(norm_w)} FRAMES)", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    step = w / len(norm_w)
    for i, val in enumerate(norm_w):
        bx = int(x + i * step)
        bw = int(step) + 1
        # Màu từ Xanh dương (ít chú ý) -> Đỏ rực (chú ý cao)
        hue = int((1.0 - val) * 120)  # 120 = Green, 0 = Red
        color = cv2.applyColorMap(np.array([[[int(val * 255)]]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        col_bgr = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(frame, (bx, y), (bx + bw, y + h), col_bgr, cv2.FILLED)


def main():
    print("=" * 60)
    print("KHỞI ĐỘNG HỆ THỐNG NHẬN DIỆN CỬ CHỈ (HETHONGXULITAY)")
    print("  - Pipeline: Đa luồng Zero-Lag (Threaded Camera -> AI Worker -> Action Dispatcher)")
    print("  - Backend: MediaPipe Hands + Dual-Pipeline (Raw Mode 1 / Kalman Mode 2)")
    print("  - Model: Multi-Scale Temporal CNN + Bi-LSTM + Temporal Self-Attention")
    print("  - Phím 'Q' hoặc ESC: Thoát | Phím 'M': Chuyển Mode")
    print("=" * 60)

    # 1. Tối ưu hóa tiến trình Windows (Priority + Tắt Power Throttling / EcoQoS)
    opt_result = optimize_windows_process()
    if opt_result.get("priority_set"):
        print("[PROCESS] Da thiet lap do uu tien ABOVE_NORMAL_PRIORITY_CLASS (0x8000).")
    if opt_result.get("throttling_disabled"):
        print("[PROCESS] Da vo hieu hoa Windows Power Throttling / EcoQoS (chong drop FPS khi chay ngam).")

    # 2. Hàng đợi sự kiện cho Action Dispatcher
    action_queue = queue.Queue(maxsize=10)
    dispatcher = ActionDispatcher(action_queue)
    dispatcher.start()

    # 3. Khởi tạo Camera đa luồng không độ trễ (Zero-Buffer-Lag Threaded Camera)
    camera = ThreadedCamera(
        src=config.CAMERA_INDEX,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
        buffersize=getattr(config, "CAMERA_BUFFERSIZE", 1),
    )

    # 3. Khởi tạo các module thị giác và AI với đầy đủ tham số cấu hình
    enhancer = CLAHEEnhancer(
        enabled=config.CLAHE_ENABLED,
        clip_limit=config.CLAHE_CLIP_LIMIT,
        grid_size=config.CLAHE_GRID_SIZE,
        auto_gamma=config.CLAHE_AUTO_GAMMA,
        anti_glare=config.CLAHE_ANTI_GLARE,
        edge_sharpen=getattr(config, "CLAHE_EDGE_SHARPEN", True),
    )
    kalman_hands = [
        HandKalmanFilter(
            grace_frames=config.KALMAN_GRACE_FRAMES,
            dt=1.0 / float(config.CAMERA_FPS),
            process_noise_var=config.KALMAN_PROCESS_NOISE_VAR,
            measurement_noise_var=config.KALMAN_MEASUREMENT_NOISE_VAR,
        ),
        HandKalmanFilter(
            grace_frames=config.KALMAN_GRACE_FRAMES,
            dt=1.0 / float(config.CAMERA_FPS),
            process_noise_var=config.KALMAN_PROCESS_NOISE_VAR,
            measurement_noise_var=config.KALMAN_MEASUREMENT_NOISE_VAR,
        ),
    ]
    smoother = VotingSmoother(
        window_size=config.VOTING_WINDOW_SIZE,
        min_ratio=config.VOTING_MIN_RATIO,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        cooldown_sec=config.ACTION_COOLDOWN_SEC,
    )

    anatomical_guard = AnatomicalGuard(
        enabled=getattr(config, "ANATOMICAL_GUARD_ENABLED", True),
        max_finger_palm_ratio=getattr(config, "MAX_FINGER_PALM_RATIO", 1.25),
        max_segment_palm_ratio=getattr(config, "MAX_SEGMENT_PALM_RATIO", 0.85),
        min_phalanx_ratio=getattr(config, "MIN_PHALANX_RATIO", 0.35),
        max_phalanx_ratio=getattr(config, "MAX_PHALANX_RATIO", 3.50),
    )

    mouse_controller = MouseController(
        frame_w=config.CAMERA_WIDTH,
        frame_h=config.CAMERA_HEIGHT,
    )
    # Đồng bộ kích thước khung hình phần cứng thực tế với mouse_controller
    try:
        actual_cam_w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_cam_h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_cam_w and actual_cam_h and int(actual_cam_w) > 0 and int(actual_cam_h) > 0:
            mouse_controller.set_frame_dimensions(int(actual_cam_w), int(actual_cam_h))
    except Exception:
        pass

    mp_detector = mp_hands.Hands(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    ) if mp_hands else None

    # 4. Nạp mô hình Deep Learning (Ưu tiên FP32 ONNX trên CPU vì nhanh hơn 1.58x so với INT8 dynamic quantize)
    ort_session = None
    torch_model = None
    engine_name = "Chưa nạp"

    # Thử nạp FP32 ONNX trước (hiệu năng cao nhất trên CPU)
    if config.MODEL_ONNX_PATH.exists():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(config.MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
            inp_dim = sess.get_inputs()[0].shape[-1]
            if isinstance(inp_dim, int) and inp_dim != config.FEATURE_DIM:
                print(f"[!] FP32 ONNX co so chieu cu ({inp_dim} != {config.FEATURE_DIM}). Bo qua.")
            else:
                ort_session = sess
                engine_name = "FP32 ONNX (CPU)"
                print(f"[+] Da nap thanh cong mo hinh FP32 ONNX: {config.MODEL_ONNX_PATH.name} (Toi uu toc do)")
        except Exception as e:
            print(f"[!] Khong the nap FP32 ONNX: {e}")

    # Fallback sang INT8 ONNX nếu không có FP32
    if ort_session is None and config.MODEL_INT8_PATH.exists():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(config.MODEL_INT8_PATH), providers=["CPUExecutionProvider"])
            inp_dim = sess.get_inputs()[0].shape[-1]
            if isinstance(inp_dim, int) and inp_dim != config.FEATURE_DIM:
                print(f"[!] INT8 ONNX co so chieu cu ({inp_dim} != {config.FEATURE_DIM}). Bo qua.")
            else:
                ort_session = sess
                engine_name = "INT8 ONNX (CPU)"
                print(f"[+] Da nap mo hinh INT8 ONNX: {config.MODEL_INT8_PATH.name}")
        except Exception as e:
            print(f"[!] Khong the nap INT8 ONNX: {e}")

    if ort_session is None:
        # Fallback về PyTorch
        device = torch.device(config.DEVICE if torch.cuda.is_available() and "cuda" in config.DEVICE else "cpu")
        torch_model = MultiScaleBiLSTMAttention(
            input_dim=config.FEATURE_DIM,
            num_classes=config.NUM_CLASSES
        ).to(device)
        if config.MODEL_PT_PATH.exists():
            try:
                state_dict = torch.load(config.MODEL_PT_PATH, map_location=device)
                torch_model.load_state_dict(state_dict)
                print(f"[+] Da nap trong so PyTorch tu: {config.MODEL_PT_PATH.name}")
            except Exception as e:
                print(f"[!] Khong the nap trong so PyTorch (co the do khac {config.FEATURE_DIM}D): {e}")
                print(f"[*] Su dung mo hinh PyTorch mac dinh. Hay chay src/training/train.py de cap nhat weights.")
        torch_model.eval()

    # Bộ đệm trượt chuỗi thời gian (Circular Buffer)
    temporal_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
    # Khởi tạo đầy SEQUENCE_LENGTH frame vector 0
    for _ in range(config.SEQUENCE_LENGTH):
        temporal_buffer.append([0.0] * config.FEATURE_DIM)

    prev_time = time.time()
    fps = 30.0
    curr_stable_label = "IDLE"
    curr_confidence = 0.0
    attn_weights = np.zeros(config.SEQUENCE_LENGTH, dtype=np.float32)
    had_hands_prev = False  # Cờ Edge-Triggered cho Failsafe Release
    current_app_mode = "MODE1_MOUSE"  # Mặc định Mode 1: Điều khiển chuột siêu mượt

    # 5. Vòng lặp AI Worker chính
    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            enhanced = enhancer.process(frame)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            # Phát hiện bàn tay & Lọc Kalman với cơ chế Chống tráo đổi tay (Anti-Swapping Matcher 1 & 2 tay)
            hands_data = []      # Landmark đã lọc Kalman cho Mode 2 (Bi-LSTM 158D Context Gate)
            raw_hands_data = []  # Landmark thô sau AnatomicalGuard cho Mode 1 (Chuột 0ms Zero-Lag)
            velocities = []
            real_hand_idx = None
            h, w, _ = frame.shape

            if mp_detector:
                results = mp_detector.process(rgb)
                detected_lms = results.multi_hand_landmarks if (results and results.multi_hand_landmarks) else []
                detected_count = len(detected_lms)

                # Thuật toán ghép cặp vị trí (Greedy Distance Association) để chống hoán đổi khi bắt chéo 2 tay hoặc hạ 1 tay
                if detected_count == 2 and kalman_hands[0].has_active_hand and kalman_hands[1].has_active_hand:
                    prev_w0 = kalman_hands[0].filters[0].x[0:2, 0]
                    prev_w1 = kalman_hands[1].filters[0].x[0:2, 0]
                    cur_w0 = np.array([detected_lms[0].landmark[0].x * w, detected_lms[0].landmark[0].y * h])
                    cur_w1 = np.array([detected_lms[1].landmark[0].x * w, detected_lms[1].landmark[0].y * h])

                    cost_direct = np.linalg.norm(cur_w0 - prev_w0) + np.linalg.norm(cur_w1 - prev_w1)
                    cost_swapped = np.linalg.norm(cur_w1 - prev_w0) + np.linalg.norm(cur_w0 - prev_w1)

                    if cost_swapped < cost_direct:
                        detected_lms = [detected_lms[1], detected_lms[0]]

                elif detected_count == 1 and kalman_hands[0].has_active_hand and kalman_hands[1].has_active_hand:
                    prev_w0 = kalman_hands[0].filters[0].x[0:2, 0]
                    prev_w1 = kalman_hands[1].filters[0].x[0:2, 0]
                    cur_w = np.array([detected_lms[0].landmark[0].x * w, detected_lms[0].landmark[0].y * h])
                    d0 = np.linalg.norm(cur_w - prev_w0)
                    d1 = np.linalg.norm(cur_w - prev_w1)
                    if d1 < d0:
                        # Tay còn lại thuộc về kalman_hands[1], tay 0 đã hạ
                        detected_lms = [None, detected_lms[0]]
                        detected_count = 2

                for h_idx in range(min(2, detected_count)):
                    hand_lms = detected_lms[h_idx]
                    if hand_lms is not None:
                        raw_pts = [(i, l.x * w, l.y * h, l.z * w) for i, l in enumerate(hand_lms.landmark)]

                        # Bộ lọc giải phẫu học sinh học (Anatomical Sanity Guard)
                        is_anatomical_valid, guard_reason = anatomical_guard.validate_hand(raw_pts)
                        if not is_anatomical_valid:
                            # Chặn đứng 100% hiện tượng kéo dãn xương lên vai/mặt
                            cv2.putText(enhanced, f"CHAN DI DANG: {guard_reason}", (15, h - 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            smoothed_pts, is_valid = kalman_hands[h_idx].update(None)
                            if is_valid:
                                hands_data.append(smoothed_pts)
                                velocities.append(kalman_hands[h_idx].get_hand_velocity())
                            continue

                        mp_drawing.draw_landmarks(enhanced, hand_lms, mp_hands.HAND_CONNECTIONS)

                        # Mode 1: Lưu trực tiếp raw_pts hợp lệ (Zero-Lag 0ms)
                        if real_hand_idx is None:
                            real_hand_idx = len(raw_hands_data)
                        raw_hands_data.append(raw_pts)

                        # Mode 2: Vẫn tiếp tục nạp vào Kalman Filter 6D để tính toán vận tốc và làm mượt cho Bi-LSTM
                        smoothed_pts, is_valid = kalman_hands[h_idx].update(raw_pts)
                        if is_valid:
                            hands_data.append(smoothed_pts)
                            velocities.append(kalman_hands[h_idx].get_hand_velocity())
                    else:
                        smoothed_pts, is_valid = kalman_hands[h_idx].update(None)
                        if is_valid:
                            hands_data.append(smoothed_pts)
                            velocities.append(kalman_hands[h_idx].get_hand_velocity())

                # Cho phép Kalman dead-reckoning cho tay bị mất dấu
                for h_idx in range(detected_count, 2):
                    smoothed_pts, is_valid = kalman_hands[h_idx].update(None)
                    if is_valid:
                        hands_data.append(smoothed_pts)
                        velocities.append(kalman_hands[h_idx].get_hand_velocity())

            # Trích xuất 158D Context Gate và đẩy vào bộ đệm trượt SEQUENCE_LENGTH frame
            mode_val = config.MODE_MAP.get(current_app_mode, 1 if current_app_mode == "MODE1_MOUSE" else 2)
            feat_158d = extract_158d_features(hands_data, velocities, mode_id=mode_val)
            temporal_buffer.append(feat_158d)

            # ── XỬ LÝ THEO CHẾ ĐỘ HIỆN TẠI (HYBRID DUAL-ENGINE) ──
            if current_app_mode == "MODE1_MOUSE":
                # Mode 1: Điều khiển chuột tức thì (Zero-Lag 0ms latency)
                # Tách riêng đường truyền: Dùng trực tiếp raw_hands_data (đã qua AnatomicalGuard),
                # kết hợp One Euro Filter (Beta = 0.04) trong MouseController. Bỏ qua hoàn toàn Kalman.
                active_hand = []
                if len(raw_hands_data) == 1:
                    active_hand = raw_hands_data[0]
                elif len(raw_hands_data) > 1:
                    # Ưu tiên bàn tay đang duỗi ngón trỏ để điều khiển chuột
                    hand_scores = []
                    for h_pts in raw_hands_data:
                        f_up = mouse_controller.count_fingers_up(h_pts)
                        # Điểm ưu tiên: ngón trỏ duỗi (f_up[1] == 1)
                        score = 2 if f_up[1] == 1 else (1 if sum(f_up) > 0 else 0)
                        hand_scores.append(score)
                    best_idx = int(np.argmax(hand_scores))
                    active_hand = raw_hands_data[best_idx]
                mouse_controller.process(enhanced, active_hand)
            else:
                # Mode 2: Nhận diện chuỗi cử chỉ thời gian (Bi-LSTM 158D Context Gate)
                # Cơ chế Failsafe Edge-Triggered: Chỉ phát tín hiệu nhả chuột DUY NHẤT 1 LẦN khi mất toàn bộ dấu tay
                if len(hands_data) == 0:
                    if had_hands_prev:
                        try:
                            action_queue.put_nowait({"action": "FAILSAFE_RELEASE", "payload": {}})
                        except queue.Full:
                            pass
                        had_hands_prev = False
                else:
                    had_hands_prev = True

                # Suy luận cử chỉ với mạng nơ-ron
                raw_class_idx = 0
                raw_conf = 0.0

                if len(temporal_buffer) == config.SEQUENCE_LENGTH and len(hands_data) > 0:
                    seq_np = np.array([temporal_buffer], dtype=np.float32)  # (1, SEQUENCE_LENGTH, 158)

                    if ort_session is not None:
                        # Chạy ONNX FP32 / INT8
                        ort_outs = ort_session.run(None, {"input_sequence": seq_np})
                        logits = ort_outs[0][0]
                        attn_weights = ort_outs[1][0] if len(ort_outs) > 1 else np.zeros(config.SEQUENCE_LENGTH)

                        # Softmax
                        exp_l = np.exp(logits - np.max(logits))
                        probs = exp_l / np.sum(exp_l)
                        raw_class_idx = int(np.argmax(probs))
                        raw_conf = float(probs[raw_class_idx])

                    elif torch_model is not None:
                        # Chạy PyTorch
                        with torch.no_grad():
                            tensor_inp = torch.tensor(seq_np, device=device)
                            logits_t, attn_t = torch_model(tensor_inp)
                            probs_t = torch.softmax(logits_t, dim=-1)[0]
                            raw_class_idx = int(torch.argmax(probs_t).item())
                            raw_conf = float(probs_t[raw_class_idx].item())
                            attn_weights = attn_t[0].cpu().numpy()

                predicted_label = config.GESTURE_CLASSES[raw_class_idx] if len(hands_data) > 0 else "IDLE"

                # Context Gate: Lọc các cử chỉ hợp lệ theo Mode đang kích hoạt
                allowed_classes = config.MODE_GESTURE_CLASSES.get(mode_val, config.GESTURE_CLASSES)
                if predicted_label in allowed_classes:
                    raw_label_name = predicted_label
                else:
                    raw_label_name = "IDLE"
                    raw_conf = 0.0

                # Làm mượt đa số và máy trạng thái Cooldown
                action_to_fire, curr_confidence, curr_stable_label = smoother.update(raw_label_name, raw_conf)

                # Nếu có hành động xác nhận -> Đẩy vào Action Dispatcher
                if action_to_fire:
                    try:
                        action_queue.put_nowait({"action": action_to_fire, "payload": {}})
                    except queue.Full:
                        pass

            # Đo FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (curr_time - prev_time)))
            prev_time = curr_time

            # ── VẼ HUD TRỰC QUAN HÓA ──
            # 1. Bảng trạng thái góc trên bên trái
            cv2.rectangle(enhanced, (15, 15), (420, 125), (20, 20, 20), cv2.FILLED)
            cv2.rectangle(enhanced, (15, 15), (420, 125), (0, 200, 255), 1)

            if current_app_mode == "MODE1_MOUSE":
                cv2.putText(enhanced, "MODE 1: CHUOT 0ms (Nhan 'M': Doi)", (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
                cv2.putText(enhanced, f"Lenh: {mouse_controller.action_text or 'Di chuot'}", (25, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(enhanced, f"Trang thai: {mouse_controller.current_mode} | FPS: {fps:.1f}", (25, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 150), 1)
            else:
                cv2.putText(enhanced, f"MODE 2: {curr_stable_label} (Nhan 'M': Doi)", (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(enhanced, f"Do tin cay: {curr_confidence*100:.1f}% | FPS: {fps:.1f}", (25, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                cv2.putText(enhanced, f"Dong co: {engine_name}", (25, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 150), 1)

                # 2. Thanh Attention Heatmap Bar góc dưới bên trái (khi ở Mode 2)
                draw_attention_heatmap(enhanced, attn_weights, x=25, y=h - 40, w=260, h=18)

            WINDOW_TITLE = "HETHONGXULITAY - AI Gesture Control"
            cv2.imshow(WINDOW_TITLE, enhanced)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass
            if key == ord('m') or key == ord('M'):
                if current_app_mode == "MODE1_MOUSE":
                    current_app_mode = "MODE2_GESTURES"
                    mouse_controller.cancel_click_and_drag()
                    mouse_controller.reset()
                    smoother.reset()
                    # Khoi tao lai bo dem thoi gian cho Mode 2 de tranh giat cu chi cu tu Mode 1
                    mode_val_target = config.MODE_MAP.get("MODE2_GESTURES", 2)
                    norm_m = float(mode_val_target) / float(getattr(config, "MAX_MODES", 4))
                    temporal_buffer.clear()
                    for _ in range(config.SEQUENCE_LENGTH):
                        temporal_buffer.append([0.0] * (config.FEATURE_DIM - 1) + [norm_m])
                    curr_stable_label = "IDLE"
                    curr_confidence = 0.0
                    attn_weights = np.zeros(config.SEQUENCE_LENGTH, dtype=np.float32)
                    had_hands_prev = False
                    print("[*] Chuyen sang MODE 2: Cu chi dong (Bi-LSTM 158D Context Gate - Buffer Reset)")
                else:
                    current_app_mode = "MODE1_MOUSE"
                    mouse_controller.reset()
                    smoother.reset()
                    # Xoa cac hanh dong cu chi dong con sot lai trong queue
                    while not action_queue.empty():
                        try:
                            action_queue.get_nowait()
                        except queue.Empty:
                            break
                    mode_val_target = config.MODE_MAP.get("MODE1_MOUSE", 1)
                    norm_m = float(mode_val_target) / float(getattr(config, "MAX_MODES", 4))
                    temporal_buffer.clear()
                    for _ in range(config.SEQUENCE_LENGTH):
                        temporal_buffer.append([0.0] * (config.FEATURE_DIM - 1) + [norm_m])
                    print("[*] Chuyen sang MODE 1: Dieu khien chuot sieu muot (Zero-Lag 0ms - Buffer Reset)")

    finally:
        print("[*] Đang đóng hệ thống và giải phóng tài nguyên...")
        mouse_controller.cancel_click_and_drag()
        mouse_controller.sys_ctrl.stop()
        camera.release()
        dispatcher.stop()
        dispatcher.join(timeout=1.5)

        if mp_detector is not None:
            try:
                mp_detector.close()
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print("[+] Đã giải phóng sạch sẽ toàn bộ luồng, camera và bộ nhớ.")


if __name__ == "__main__":
    main()
