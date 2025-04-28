from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json
import pandas as pd
from datetime import datetime
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_capture  # Import directly since camera_capture.py is in the same directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_FILE = os.path.join(BASE_DIR, 'attendance_log.txt')
EXCEL_FILE = os.path.join(BASE_DIR, 'attendance_log.xlsx')
CAPTURED_IMAGES_DIR = os.path.join(BASE_DIR, 'captured_images')

import pandas as pd

def read_attendance():
    if not os.path.exists(EXCEL_FILE):
        return []
    try:
        df = pd.read_excel(EXCEL_FILE)
        attendance = []
        for _, row in df.iterrows():
            attendance.append({
                'date': str(row.get('Date', '')),
                'time': str(row.get('Time', '')),
                'subject': str(row.get('Subject', '')),
                'name': str(row.get('Name', '')),
                'roll_number': str(row.get('Roll Number', ''))
            })
        # Do not clear the Excel file after reading to preserve all logs
        # empty_df = pd.DataFrame(columns=df.columns)
        # empty_df.to_excel(EXCEL_FILE, index=False)
        return attendance
    except Exception as e:
        print(f"Error reading Excel attendance file: {e}")
        return []

@app.route('/')
def index():
    attendance = read_attendance()
    return render_template('index.html', attendance=attendance)

@app.route('/api/attendance')
def api_attendance():
    attendance = read_attendance()
    return jsonify(attendance)

@app.route('/captured_images/<filename>')
def captured_image(filename):
    return send_from_directory(CAPTURED_IMAGES_DIR, filename)

@app.route('/capture', methods=['POST'])
def capture():
    # Clear captured_images folder before capturing new image
    for f in os.listdir(CAPTURED_IMAGES_DIR):
        file_path = os.path.join(CAPTURED_IMAGES_DIR, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    filename = camera_capture.capture_image(save_path=CAPTURED_IMAGES_DIR, delay=3)
    if filename:
        rel_path = os.path.relpath(filename, BASE_DIR).replace("\\", "/")
        # Run recognition immediately on the captured image
        subject = 'Unknown'  # or get from request if needed
        image_filename = os.path.basename(filename)
        try:
            python_executable = sys.executable
            result = subprocess.run(
                [python_executable, 'recognize_and_log.py', subject, image_filename],
                capture_output=True, text=True, check=True,
                cwd=BASE_DIR
            )
            # Extract JSON part from stdout by finding first '[' and last ']'
            stdout = result.stdout
            start = stdout.find('[')
            end = stdout.rfind(']') + 1
            json_str = stdout[start:end] if start != -1 and end != -1 else '[]'
            recognition_results = json.loads(json_str)
        except Exception as e:
            print(f"Recognition subprocess error: {e}")
            recognition_results = []

        return jsonify({
            'status': 'success',
            'message': 'Image captured and attendance recognized.',
            'image_path': '/' + rel_path,
            'recognition_results': recognition_results
        })
    else:
        return jsonify({'status': 'error', 'message': 'Failed to capture image.'}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    subject = data.get('subject', 'Unknown') if data else 'Unknown'
    image_path = data.get('image_path') if data else None
    image_filename = None
    if image_path:
        image_filename = os.path.basename(image_path)
    try:
        python_executable = sys.executable
        cmd = [python_executable, 'recognize_and_log.py', subject]
        if image_filename:
            cmd.append(image_filename)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, check=True,
            cwd=BASE_DIR
        )
        print("Subprocess stdout:", repr(result.stdout))  # Debug print with repr
        print("Subprocess stderr:", repr(result.stderr))  # Debug print with repr
        # Extract JSON part from stdout by finding first '[' and last ']'
        stdout = result.stdout
        # Remove ANSI escape sequences from stdout
        import re
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        clean_stdout = ansi_escape.sub('', stdout)
        start = clean_stdout.find('[')
        end = clean_stdout.rfind(']') + 1
        json_str = clean_stdout[start:end] if start != -1 and end != -1 else '[]'
        recognition_results = json.loads(json_str)
        return jsonify({'status': 'success', 'message': 'Attendance recognized and logged.', 'results': recognition_results})
    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e.stderr)  # Debug print
        return jsonify({'status': 'error', 'message': 'Failed to recognize and log attendance.', 'error': e.stderr}), 500
    except json.JSONDecodeError:
        print("JSON decode error. Subprocess output:", result.stdout)  # Debug print
        return jsonify({'status': 'error', 'message': 'Failed to parse recognition results.'}), 500
    except Exception as e:
        print(f"Unexpected error in recognize route: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
