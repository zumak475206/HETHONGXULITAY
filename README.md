# HETHONGXULITAY — AI Hand Gesture Studio & Control Center

Hệ thống điều khiển máy tính và nhận diện cử chỉ bàn tay thời gian thực độ trễ siêu thấp (Zero-Lag 0ms) tích hợp mô hình học sâu chuỗi thời gian (Bi-LSTM + Attention) và ứng dụng Desktop PyQt6 hiện đại.

---

## 🌟 Tính Năng Nổi Bật

- **Zero-Lag Dual-Pipeline**: Phân tách luồng điều khiển chuột (0ms latency, lọc thích ứng vận tốc qua One Euro Filter) và luồng nhận diện chuỗi cử chỉ nơ-ron (Bi-LSTM 158D).
- **Chuột Công Thái Học Mode 1**:
  - Di chuột mượt mà, khóa vị trí khi bấm chuột (Pinch Position Lock).
  - Máy trạng thái kéo thả (Drag & Drop) thông minh.
  - V-Sign Hold sạc 0.35s kích hoạt Click Phải với vòng tròn tiến trình Radial.
  - Cần gạt cuộn trang đàn hồi (Anchor Joystick Scroll).
  - Phanh nắm đấm tạm dừng (Fist Lock) chống bấm nhầm.
- **Tiền Xử Lý Ảnh Chuyên Sâu**:
  - Không gian màu LAB phân tách độc lập độ sáng.
  - Gaussian Blur khử nhiễu sensor trước khi cân bằng biểu đồ tương phản cục bộ (CLAHE).
  - Unsharp Masking làm rõ viền da tay.
  - Bảng tra 256-LUT Anti-Glare nén mềm các vùng cháy sáng.
- **Bộ Lọc Giải Phẫu Sinh Học (Anatomical Guard)**: Ngăn chặn 100% hiện tượng MediaPipe kéo dãn mốc xương lên vai/ngực khi tiếp xúc da.
- **Mạng Học Sâu Chuỗi Thời Gian**:
  - Tầng Conv1D đa thang đo (Kernel 3, 7, 15) bắt các vi chuyển động, vuốt tay và xoay vòng.
  - Mạng Bi-LSTM 2 lớp (hidden size 128).
  - Cơ chế tự chú ý (Temporal Self-Attention) trích xuất Attention Heatmap thời gian thực.
- **Ứng Dụng Desktop PyQt6 Hiện Đại**:
  - Trang Chủ: Camera Canvas 60 FPS, HUD thông số thời gian thực, dải chuyển Mode kép (nút bấm hoặc xòe 2 tay 1.8s).
  - Cửa sổ Mini Floating PiP ghim góc màn hình.
  - Kho Cử Chỉ & Thống Kê: Card trực quan, cơ sở dữ liệu SQLite `gestures.db`.
  - Studio Thu Thập & Huấn Luyện AI: Hộp thoại đếm ngược 3-2-1, gom 45 frames 158D, xuất/nhập gói ZIP phân tán, Huấn luyện AI Hot-Reload không cần tắt app.
  - Cài Đặt Hệ Thống: Điều chỉnh độ nhạy chuột, Y-Gain gập cổ tay, thông số One Euro Filter và đồng bộ thời gian thực.

---

## 🚀 Cài Đặt & Khởi Chạy

### 1. Yêu cầu môi trường
- Python 3.10+
- Webcam chuẩn (720p hoặc 480p)
- Hệ điều hành: Windows 10/11

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```
*(Hoặc cài đặt các gói cốt lõi: `opencv-python mediapipe PyQt6 torch torchvision pyautogui pycaw screen_brightness_control`)*

### 3. Khởi chạy ứng dụng
```bash
python app.py
```

### 4. Chạy kiểm thử tự động
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📁 Cấu Trúc Dự Án

```
HETHONGXULITAY/
├── app.py                     # Điểm khởi chạy chính ứng dụng PyQt6
├── config.py                  # Cấu hình tham số toàn cục
├── src/
│   ├── actions/               # Bộ thực thi hành động hệ thống
│   ├── database/              # SQLite Manager & Bộ ghép dữ liệu phân tán .npz
│   ├── features/              # Trích xuất vector đặc trưng 158D Context Gate
│   ├── inference/             # Bộ bình chọn đa số VotingSmoother
│   ├── models/                # Kiến trúc Multi-Scale 1D-CNN + Bi-LSTM + Attention
│   ├── modes/                 # Logic điều khiển chuột Mode 1 công thái học
│   ├── preprocessing/         # CLAHE Enhancer & Gaussian Blur
│   ├── tracking/              # ThreadedCamera, One Euro Filter, Kalman, Anatomical Guard
│   └── ui/                    # Giao diện PyQt6 (MainWindow, Tabs, Workers, PiP, Theme)
├── dataset/                   # Cơ sở dữ liệu SQLite & chuỗi mẫu .npz
├── docs/                      # Tài liệu báo cáo đồ án Word (.docx) & script build
├── tests/                     # 119 bài kiểm thử tự động toàn diện
└── weights/                   # Trọng số mô hình AI (PyTorch .pt, ONNX, INT8)
```
