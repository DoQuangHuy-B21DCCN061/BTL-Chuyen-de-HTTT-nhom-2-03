from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import numpy as np
import logging
from deepface import DeepFace
import json
import base64
from PIL import Image
import io
from queue import Queue
import concurrent.futures
import os
import uuid
import face_recognition
import pickle
from imutils import paths

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Biến toàn cục
camera = None
camera_lock = threading.Lock()
is_camera_running = False
frame_count = 0
last_emotion_detection = time.time()
EMOTION_DETECTION_INTERVAL = 0.3  # Giảm interval xuống 0.3 giây
fps = 0
fps_start_time = time.time()
fps_frame_count = 0
last_face_result = None
last_face_time = 0
FACE_HOLD_TIME = 0.5  # Giảm thời gian giữ bounding box xuống 0.5 giây
frame_queue = Queue(maxsize=2)  # Queue để lưu frame
result_queue = Queue(maxsize=2)  # Queue để lưu kết quả nhận diện
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Load face encodings
with open('encodings.pickle', 'rb') as f:
    face_data = pickle.load(f)

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 60)  # Tăng FPS camera lên 60
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer size
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

def calculate_fps():
    global fps, fps_frame_count, fps_start_time
    fps_frame_count += 1
    if time.time() - fps_start_time > 1.0:
        fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()
    return fps

def detect_emotion(frame):
    try:
        results = DeepFace.analyze(
            frame, 
            actions=["emotion"],
            enforce_detection=False,
            detector_backend='opencv'
        )
        return results if results and isinstance(results, list) and len(results) > 0 else None
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
        return None

def process_frame(frame):
    global frame_count, last_emotion_detection, last_face_result, last_face_time
    
    # Tính FPS
    current_fps = calculate_fps()
    
    # Resize frame để tăng hiệu năng
    frame = cv2.resize(frame, (320, 240))  # Giảm kích thước frame
    
    # Chỉ nhận diện cảm xúc mỗi EMOTION_DETECTION_INTERVAL giây
    current_time = time.time()
    if current_time - last_emotion_detection >= EMOTION_DETECTION_INTERVAL:
        # Submit task vào thread pool
        future = executor.submit(detect_emotion, frame)
        try:
            results = future.result(timeout=0.2)  # Timeout 0.2 giây
            if results:
                last_face_result = results
                last_face_time = current_time
            last_emotion_detection = current_time
        except concurrent.futures.TimeoutError:
            logger.warning("Emotion detection timeout")
    
    # Vẽ bounding box và cảm xúc từ kết quả gần nhất nếu còn trong thời gian giữ
    if last_face_result and (current_time - last_face_time <= FACE_HOLD_TIME):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_boxes = []
        # Collect face regions from emotion detection
        for res in last_face_result:
            region = res.get("region", {})
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            # Convert to (top, right, bottom, left) for face_recognition
            top, right, bottom, left = y, x + w, y + h, x
            face_boxes.append((top, right, bottom, left))
        # Get encodings for detected faces
        encodings = face_recognition.face_encodings(rgb_frame, face_boxes)
        for i, res in enumerate(last_face_result):
            region = res.get("region", {})
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            emotion = res.get("dominant_emotion", "")
            name = "Unknown"
            if i < len(encodings):
                matches = face_recognition.compare_faces(face_data["encodings"], encodings[i], tolerance=0.4)
                face_distances = face_recognition.face_distance(face_data["encodings"], encodings[i])
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = face_data["names"][best_match_index]
            label = f"{name} ({emotion})" if emotion else name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def camera_thread():
    global is_camera_running
    while is_camera_running:
        try:
            camera = get_camera()
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            # Lưu frame gốc (chưa vẽ)
            raw_frame = cv2.resize(frame, (320, 240))
            _, raw_buffer = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            raw_frame_base64 = base64.b64encode(raw_buffer).decode('utf-8')

            # Xử lý frame (vẽ bounding box, nhãn, FPS)
            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Gửi cả hai frame qua WebSocket
            socketio.emit('frame', {'image': frame_base64, 'raw_image': raw_frame_base64})

        except Exception as e:
            logger.error(f"Error in camera thread: {str(e)}")
            break
    release_camera()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    global is_camera_running
    if not is_camera_running:
        is_camera_running = True
        threading.Thread(target=camera_thread, daemon=True).start()
        emit('camera_status', {'status': 'started'})

@socketio.on('stop_camera')
def handle_stop_camera():
    global is_camera_running
    is_camera_running = False
    release_camera()
    emit('camera_status', {'status': 'stopped'})

@app.route('/list_folders', methods=['GET'])
def list_folders():
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    folders = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    return jsonify({'folders': folders})

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    folder = data.get('folder')
    image_data = data.get('image')
    if not folder or not image_data:
        return jsonify({'success': False, 'message': 'Missing folder or image data'}), 400
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    folder_path = os.path.join(dataset_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Giải mã base64 và lưu ảnh
    try:
        img_bytes = base64.b64decode(image_data.split(',')[-1])
        img = Image.open(io.BytesIO(img_bytes))
        filename = f"{uuid.uuid4().hex}.jpg"
        img.save(os.path.join(folder_path, filename))
        return jsonify({'success': True, 'message': 'Image saved successfully'})
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return jsonify({'success': False, 'message': 'Error saving image'}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Lấy paths của images trong dataset
        dataset_dir = os.path.join(os.getcwd(), 'dataset')
        imagePaths = list(paths.list_images(dataset_dir))

        # Khởi tạo list chứa known encodings và known names
        knownEncodings = []
        knownNames = []

        # Duyệt qua các image paths
        for (i, imagePath) in enumerate(imagePaths):
            # Lấy tên người từ imagepath
            name = imagePath.split(os.path.sep)[-2]

            # Load image bằng OpenCV và chuyển từ BGR to RGB
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces và tính encodings
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            # Lưu encodings và names
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

        # Lưu encodings và names vào file
        data = {"encodings": knownEncodings, "names": knownNames}
        with open('encodings.pickle', 'wb') as f:
            f.write(pickle.dumps(data))

        # Reload face data
        global face_data
        with open('encodings.pickle', 'rb') as f:
            face_data = pickle.load(f)

        return jsonify({'success': True, 'message': 'Model trained successfully'})
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'success': False, 'message': f'Error training model: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
    finally:
        release_camera() 