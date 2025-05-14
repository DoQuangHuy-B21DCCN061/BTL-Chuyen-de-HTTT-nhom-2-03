# Face Recognition Web Application

A web-based face recognition and emotion detection application built with Flask, OpenCV, and deep learning.

## Features

- Real-time face recognition
- Emotion detection
- Web interface for capturing and managing face images
- Easy training of face recognition model

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the emotion detection model:
```bash
# The model will be downloaded automatically when first running the application
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Using the application:
   - Click "Start Camera" to begin video feed
   - Click "Take Photo" to capture a face image
   - Select or create a new person's folder to save the image
   - Click "Train Model" after adding new face images
   - The application will automatically recognize faces and detect emotions

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `dataset/` - Directory for storing face images
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.7+
- OpenCV
- Flask
- face_recognition
- deepface
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 