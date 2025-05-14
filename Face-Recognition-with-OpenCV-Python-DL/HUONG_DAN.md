# Ứng dụng Nhận diện Khuôn mặt và Cảm xúc

Đây là ứng dụng web sử dụng Flask và DeepFace để nhận diện khuôn mặt và cảm xúc thông qua webcam.

## Yêu cầu hệ thống

- Python 3.8 hoặc mới hơn
- Webcam hoạt động
- Trình duyệt web hiện đại (Chrome, Firefox, Edge)

## Các bước cài đặt

### 1. Cài đặt Python
- Tải và cài đặt Python từ [python.org](https://www.python.org/downloads/)
- Đảm bảo tích chọn "Add Python to PATH" khi cài đặt

### 2. Tạo môi trường ảo (Virtual Environment)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate
```

### 3. Cài đặt các thư viện
```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng
```bash
python app.py
```

### 5. Truy cập web app
- Mở trình duyệt web
- Truy cập địa chỉ: `http://localhost:5000`

## Xử lý lỗi thường gặp

### 1. Lỗi "No module named 'cv2'"
```bash
pip uninstall opencv-python
pip install opencv-python
```

### 2. Lỗi với Flask-SocketIO
```bash
pip install flask-socketio==5.1.1
```

### 3. Lỗi với DeepFace
```bash
pip install tensorflow
```

### 4. Webcam không hoạt động
- Kiểm tra quyền truy cập webcam trong trình duyệt
- Thử đóng và mở lại trình duyệt
- Kiểm tra xem webcam có đang được sử dụng bởi ứng dụng khác không

## Lưu ý quan trọng

- Đảm bảo máy tính có webcam hoạt động
- Cho phép trình duyệt truy cập webcam khi được yêu cầu
- Nếu gặp lỗi khi cài đặt OpenCV, bạn có thể thử:
  ```bash
  pip uninstall opencv-python
  pip install opencv-python-headless
  ```

## Theo dõi và gỡ lỗi

- Kiểm tra terminal để xem các thông báo lỗi
- Đảm bảo tất cả các thư viện đã được cài đặt đúng phiên bản
- Kiểm tra kết nối webcam trong ứng dụng Camera của Windows