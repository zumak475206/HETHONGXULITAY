# 📚 SỔ TAY KỸ THUẬT: XÂY DỰNG HỆ THỐNG NHẬN DIỆN CỬ CHỈ BÀN TAY TỪ CON SỐ 0
## Dự án: HETHONGXULITAY (Multi-Scale Temporal CNN + Bi-LSTM + Self-Attention)

> **Mục tiêu sư phạm**: Nắm trọn vẹn bản chất toán học, kiến trúc Deep Learning, kỹ thuật xử lý tín hiệu và hệ thống đa luồng thời gian thực mà không phụ thuộc vào các thư viện nhận diện cử chỉ "hộp đen".

---

## MỤC LỤC
1. [Kiến trúc Đa luồng 3 Tầng (3-Thread System Architecture)](#1-kiến-trúc-đa-luồng-3-tầng)
2. [Toán học Bộ lọc Động học Kalman (Kalman Filter Kinematics)](#2-toán-học-bộ-lọc-động-học-kalman)
3. [Vector Đặc trưng 157D & Canonical Hand Alignment 360°](#3-vector-đặc-trưng-157d--canonical-alignment)
4. [Mạng Nơ-ron Multi-Scale Temporal CNN + Bi-LSTM + Self-Attention](#4-kiến-trúc-mạng-deep-learning)
5. [Lượng tử hóa Mô hình INT8 Động (Dynamic Quantization)](#5-lượng-tử-hóa-mô-hình-int8)
6. [Hậu xử lý: 5-Frame Majority Voting & Cooldown State Machine](#6-hậu-xử-lý-voting--cooldown)
7. [Hướng dẫn Thực hành Từng bước (Step-by-Step Hands-on)](#7-hướng-dẫn-thực-hành-từng-bước)

---

## 1. Kiến trúc Đa luồng 3 Tầng

Trong các ứng dụng thị giác máy tính thông thường, người mới bắt đầu thường viết tất cả trong một vòng lặp `while True`:
```python
# CÁCH LÀM SAI (Gây lag và sụt giảm FPS nghiêm trọng):
while True:
    frame = cap.read()        # 1. Đợi camera (mất ~15-30ms)
    lms = detector.process()  # 2. Chạy AI (mất ~20-40ms)
    pyautogui.click()         # 3. Lời gọi OS chặn luồng (mất ~5-20ms)
    # Tổng thời gian: 40-90ms -> FPS rớt xuống 10-15 FPS!
```

### Giải pháp Công nghiệp: 3 Luồng Độc lập (Producer - Worker - Consumer)
Hệ thống `HETHONGXULITAY` tách biệt 3 tác vụ bằng các hàng đợi an toàn đa luồng (`queue.Queue`):

```
[CameraGrabber Thread]  ──(frame_queue)──>  [AI Worker Thread]  ──(action_queue)──>  [ActionDispatcher Thread]
(Luôn giữ frame mới nhất)                    (Chạy CLAHE + Kalman                      (Thực thi lệnh OS ngầm,
                                              + 157D + Bi-LSTM)                         không làm chậm AI)
```
- **Zero-Lag Dropping**: Hàng đợi `frame_queue` chỉ có kích thước `maxsize=2`. Nếu luồng AI bận, camera tự động hủy bỏ frame cũ để frame kế tiếp xử lý luôn là hình ảnh mới nhất tại thời điểm thực.

---

## 2. Toán học Bộ lọc Động học Kalman

Khi bàn tay di chuyển nhanh hoặc bị che khuất một phần, tọa độ 21 điểm trả về từ camera thường bị rung lắc (jitter) hoặc mất dấu đột ngột trong 1-2 frame. Bộ lọc Kalman giải quyết vấn đề này bằng mô hình động học vận tốc không đổi (Constant Velocity Model).

### Vector Trạng thái (State Vector)
$$x = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix}$$
Trong đó $(p_x, p_y)$ là tọa độ vị trí pixel, $(v_x, v_y)$ là vận tốc di chuyển tính bằng pixel/frame.

### 1. Bước Dự báo (Predict)
Dựa vào định luật chuyển động $p(t + \Delta t) = p(t) + v(t)\Delta t$:
$$x_{k|k-1} = F x_{k-1|k-1}$$
Ma trận chuyển trạng thái $F$:
$$F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
Hiệp phương sai sai số dự báo $P_{k|k-1}$:
$$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$
($Q$ là ma trận nhiễu mô hình gia tốc).

### 2. Bước Cập nhật (Update khi có quan sát từ Camera)
Webcam quan sát được $z_k = [z_x, z_y]^T$ thông qua ma trận quan sát $H$:
$$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
Độ lợi Kalman (Kalman Gain) $K_k$:
$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$
Cập nhật trạng thái và hiệp phương sai:
$$x_{k|k} = x_{k|k-1} + K_k (z_k - H x_{k|k-1})$$
$$P_{k|k} = (I - K_k H) P_{k|k-1}$$

> **Ý nghĩa cốt lõi**:
> - Khi camera cho kết quả nhiễu ($R$ lớn): $K_k$ nhỏ $\rightarrow$ Hệ thống tin tưởng vào quán tính dự báo của phương trình chuyển động $F x$.
> - Khi MediaPipe mất dấu 1-2 frame: Ta bỏ qua bước Update, chỉ chạy bước Predict (**Dead-Reckoning**). Tọa độ bàn tay vẫn lướt mượt mà không bị đứt đoạn!

---

## 3. Vector Đặc trưng 158D & Canonical Hand Alignment 360° (Context Gate)

Để mô hình nhận diện chính xác dù bàn tay xoay ngang, nghiêng hay chúc ngược, toàn bộ 21 điểm được quy đổi về **Hệ quy chiếu chuẩn thẳng đứng (Canonical Frame)** và bổ sung trục Z độ sâu MediaPipe:

```
      MIDDLE_MCP (9)
           ▲
           │ Vector trục bàn tay v_palm
           │ (Góc lệch theta so với 0° Up)
           │
       WRIST (0)
```

1. **Góc lệch trục**: $\theta = \text{atan2}(v_y, v_x) + \frac{\pi}{2}$.
2. **Phép biến đổi Affine Xoay ngược $-\theta$ quanh Cổ tay (XY) & Chuẩn hóa độ sâu Z**:
   $$\begin{bmatrix} x_{can} \\ y_{can} \end{bmatrix} = \frac{1}{\text{PalmSize}} \begin{bmatrix} \cos(-\theta) & -\sin(-\theta) \\ \sin(-\theta) & \cos(-\theta) \end{bmatrix} \begin{bmatrix} x - x_{wrist} \\ y - y_{wrist} \end{bmatrix}, \quad z_{can} = \frac{z - z_{wrist}}{\text{PalmSize}}$$
3. **10 Góc Uốn Khớp Nội tại 3D (Joint Flexion Angles in 3D)**:
   Góc giữa 3 điểm khớp ngón trong không gian 3 chiều $(x, y, z)$:
   $$\cos(\phi) = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|}$$
   Các góc gập này là đặc trưng nội tại của bàn tay, **bất biến toán học 100%** trước mọi góc xoay $0^\circ \to 360^\circ$ và phân biệt chính xác trạng thái ngón gập/duỗi trong chiều sâu.
4. **Cấu trúc 158 chiều (kèm Context Gate)**:
   - Tay 1: 63D tọa độ 3D xoay chuẩn ($21 \times 3$) + 10D góc đốt ngón 3D + 3D vận tốc ($v_x, v_y, v_z$) = 76D.
   - Tay 2: 76D tương tự (zeros nếu chỉ có 1 tay).
   - Quan hệ 2 tay: 5D (khoảng cách 3D 2 cổ tay, $\Delta x$, $\Delta y$, $\Delta z$, góc lệch tương đối giữa 2 lòng bàn tay).
   - **Context Gate**: 1D giá trị Mode ID chuẩn hóa $\in [0.0, 1.0]$ ($\text{mode\_id} / \text{MAX\_MODES}$) giúp mạng tự học phân biệt chế độ chuột vs cử chỉ đa phương tiện.
   - Tổng: $76 + 76 + 5 + 1 = 158\text{D}$.

---

## 4. Kiến trúc Mạng Deep Learning

```
Đầu vào: Tensor (Batch, 45 frames, 158 đặc trưng)
     │
     ▼
┌────────────────────────────────────────────────────────┐
│ Multi-Scale Temporal CNN                               │
│ - Nhánh 1 (Conv1D k=3)  -> Bắt vi cử chỉ nhanh (Click) │
│ - Nhánh 2 (Conv1D k=7)  -> Bắt cử chỉ vừa (Vuốt)       │
│ - Nhánh 3 (Conv1D k=15) -> Bắt quỹ đạo lớn (Xoay tròn) │
│ Ghép nối (Concatenate) -> Tensor (Batch, 45, 96)       │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Bi-LSTM (2 layers, hidden=128, bidirectional=True)     │
│ Đầu ra: Tensor (Batch, 45, 256)                        │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Temporal Self-Attention                                │
│ alpha_t = softmax(v^T * tanh(W * h_t + b))             │
│ Context vector c = sum(alpha_t * h_t) -> (Batch, 256)   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Classification Head (Dense -> LayerNorm -> ReLU -> Out)│
│ Đầu ra: Logits (Batch, num_classes) & Attention Weights│
└────────────────────────────────────────────────────────┘
```

---

## 5. Lượng tử hóa Mô hình INT8 Động (Dynamic Quantization)

Sau khi huấn luyện xong mô hình PyTorch FP32 (dùng số thực 32-bit `float32`), ta chuyển đổi mô hình sang `ONNX` và lượng tử hóa sang `INT8` (số nguyên 8-bit `int8`):
- **Nguyên lý**: Ánh xạ dải giá trị trọng số từ `[-W_max, W_max]` sang `[-128, 127]` theo hệ số tỷ lệ $S$:
  $$W_{int8} = \text{round}\left(\frac{W_{fp32}}{S}\right)$$
- **Lợi ích**:
  1. Kích thước file model giảm ~4 lần (từ ~1.5 MB xuống ~380 KB).
  2. Tốc độ suy luận CPU tăng ~1.8x – 2.5x nhờ tập lệnh SIMD (VNNI / AVX2).
  3. Độ chính xác suy luận (Accuracy) gần như giữ nguyên (độ lệch < 0.3%).

---

## 6. Hậu xử lý: Voting & Cooldown

- **5-Frame Majority Voting**: Gom kết quả dự đoán của 5 frame gần nhất. Cử chỉ chỉ được chấp nhận nếu có $\ge 3/5$ frame ($60\%$) đồng thuận và xác suất Softmax $\ge 80\%$.
- **Cooldown Lockout (0.6s)**: Với các cử chỉ rời rạc (Vuốt, Chụm ngón, Nắm tay), ngay sau khi kích hoạt hành động, hệ thống khóa đệm 0.6 giây để người dùng đưa tay về vị trí nghỉ mà không bị kích hoạt lặp 10 lần liên tiếp.

---

## 7. Hướng dẫn Thực hành Từng bước

### Bước 1: Kiểm thử toàn bộ các Module Toán học
Chạy bộ kiểm tra tự động để xác nhận các phương trình toán học và logic:
```powershell
& "D:\THAI_ANH_KY_WORKSPACE\Projects\HeThongXuLiBangTay\venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py"
```
*Toàn bộ 8 bài test sẽ kiểm tra CLAHE, Kalman, Invariance 360°, Model, Voting, Latency, Lighting và Fast Motion.*

### Bước 2: Tự Quay Dữ liệu Cử chỉ Thực tế
Khởi động công cụ thu thập tương tác:
```powershell
& "D:\THAI_ANH_KY_WORKSPACE\Projects\HeThongXuLiBangTay\venv\Scripts\python.exe" tools/sequence_recorder.py
```
- Bấm số `1` để chọn `PINCH_CLICK`, bấm `4` để chọn `SWIPE_LEFT`, v.v.
- Bấm `R`: Chờ 3-2-1 rồi thực hiện động tác trong 1 giây (đúng 30 frame).
- Bấm `Space`: Xem lại chuỗi vừa quay. Nếu ưng ý bấm `S` để lưu, nếu lỗi bấm `Z` để xóa.

### Bước 3: Huấn luyện Mạng Nơ-ron trên GPU RTX 3050 & Lượng tử hóa INT8
Chạy script huấn luyện tự động:
```powershell
& "D:\THAI_ANH_KY_WORKSPACE\Projects\HeThongXuLiBangTay\venv\Scripts\python.exe" src/training/train.py
```
*Script sẽ tự động train 25-30 epochs, xuất file ONNX FP32 và tự động lượng tử hóa INT8 ra `weights/gesture_bilstm_int8.onnx`.*

### Bước 4: Khởi chạy Ứng dụng Chính Đa luồng
Khởi động hệ thống điều khiển hoàn chỉnh:
```powershell
& "D:\THAI_ANH_KY_WORKSPACE\Projects\HeThongXuLiBangTay\venv\Scripts\python.exe" main.py
```
*Bạn sẽ nhìn thấy thanh Attention Heatmap Bar chuyển màu trực quan theo từng cử chỉ tay!*
