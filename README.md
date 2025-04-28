# Smart Attendance System Using CNN and Laptop Local Camera

## Problem Statement
Manual attendance in university classes is time-consuming and error-prone. An automated system using facial recognition can streamline the process.

## Objectives
1. Use a pre-trained CNN (MobileNet) to recognize student faces from a small dataset.
2. Deploy a laptop local camera to capture images at the classroom door.
3. Log attendance on a simple web page or text file.

## Project Structure
- `app.py`: Flask web app to display attendance.
- `camera_capture.py`: Script to capture images from the laptop camera.
- `model_training.py`: Script to fine-tune MobileNet for facial recognition.
- `attendance_log.txt`: Text file to log attendance.
- `templates/index.html`: HTML template for attendance display.
- `requirements.txt`: Python dependencies.

## Setup Instructions
1. Create a Python virtual environment and activate it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare your dataset folder with subfolders for each student containing their face images.
4. Train the model:
   ```
   python model_training.py
   ```
5. Use the camera capture script to capture images:
   ```
   python camera_capture.py
   ```
6. Run the Flask app:
   ```
   python app.py
   ```
7. Open your browser at `http://127.0.0.1:5000` to view attendance.

## Notes
- The current system logs attendance to a text file.
- Model training uses transfer learning on MobileNet.
- Camera capture saves images locally for processing.
- Further integration needed to connect recognition results to attendance logging.

## License
MIT License
