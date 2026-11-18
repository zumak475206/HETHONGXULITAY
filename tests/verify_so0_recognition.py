import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("D:/THAI_ANH_KY_WORKSPACE/Projects/HETHONGXULITAY")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from src.modes.mode3_sign_language import SignLanguageController
from src.ui.workers import CameraWorker

def test_o_shape_and_mode3():
    print("=== TEST 1: Kiểm tra tạo hình chữ O / Số 0 với CameraWorker.is_hand_fisted ===")
    # Tạo dữ liệu giả lập bàn tay số 0 (ngón cái chạm ngón trỏ tạo vòng tròn O, 3 ngón còn lại co)
    lm = {i: [0.5, 0.5, 0.0] for i in range(21)}
    lm[0] = [0.5, 0.8, 0.0]   # Cổ tay (wrist)
    lm[9] = [0.5, 0.5, 0.0]   # Gốc ngón giữa (MCP), palm_size = 0.30

    # Đầu ngón cái (4) và đầu ngón trỏ (8) khép sát nhau (< 0.45 palm_size)
    lm[4] = [0.49, 0.40, 0.0]
    lm[8] = [0.51, 0.40, 0.0]

    # Các ngón giữa, nhẫn, út co chặt (d_tip < d_pip)
    lm[10] = [0.50, 0.45, 0.0]; lm[12] = [0.50, 0.55, 0.0]  # ngón giữa co
    lm[14] = [0.55, 0.45, 0.0]; lm[16] = [0.55, 0.55, 0.0]  # ngón nhẫn co
    lm[18] = [0.60, 0.45, 0.0]; lm[20] = [0.60, 0.55, 0.0]  # ngón út co

    hand_pts = [(i, lm[i][0], lm[i][1], lm[i][2]) for i in range(21)]

    # 1. is_hand_fisted phải trả về False
    is_fist = CameraWorker.is_hand_fisted(hand_pts)
    print(f" -> CameraWorker.is_hand_fisted: {is_fist} (Kỳ vọng: False)")
    assert not is_fist, "Lỗi: Số 0 không được coi là nắm đấm!"

    # 2. SignLanguageController phải phát hiện is_o_shape = True
    print("\n=== TEST 2: Kiểm tra SignLanguageController.get_finger_extension_state ===")
    sc = SignLanguageController()
    f_state = sc.get_finger_extension_state(hand_pts)
    is_o = f_state.get("is_o_shape", False)
    d_norm = f_state.get("d_thumb_index_norm", 999)
    print(f" -> d_thumb_index_norm: {d_norm:.3f}")
    print(f" -> is_o_shape: {is_o} (Kỳ vọng: True)")
    assert is_o, "Lỗi: get_finger_extension_state không nhận diện được is_o_shape!"

    # 3. Quá trình suy luận process_inference
    print("\n=== TEST 3: Kiểm tra suy luận process_inference với bias nghiêng về số 1 ===")
    num_classes = len(config.GESTURE_CLASSES)
    logits = np.zeros(num_classes, dtype=np.float32)
    # Giả lập model thô đang bị thiên vị nhãn VSL_SO1
    if "VSL_SO1" in config.GESTURE_CLASSES:
        logits[config.GESTURE_CLASSES.index("VSL_SO1")] = 3.5

    lbl, conf, sentence = sc.process_inference(
        logits=logits,
        attn_weights=np.zeros(config.SEQUENCE_LENGTH, dtype=np.float32),
        class_list=config.GESTURE_CLASSES,
        has_hands=True,
        raw_hands_data=[hand_pts],
    )
    print(f" -> Nhãn suy luận: {lbl}")
    print(f" -> Độ tin cậy: {conf*100:.1f}%")
    print(f" -> Top 3 dự đoán: {sc.top3_predictions}")
    assert lbl == "VSL_SO0", f"Kỳ vọng VSL_SO0 nhưng nhận được {lbl}"
    assert conf >= 0.85, f"Kỳ vọng conf >= 85%, thực tế: {conf*100:.1f}%"

    print("\n=== TẤT CẢ CÁC BƯỚC KIỂM TRA ĐỀU THÀNH CÔNG VƯỢT TRỘI! ===")

if __name__ == "__main__":
    test_o_shape_and_mode3()
