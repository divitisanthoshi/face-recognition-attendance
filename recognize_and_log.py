import os
import sys
import numpy as np
import cv2
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import json
import os
import datetime
import pandas as pd

def log_attendance(name, subject):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"{timestamp} - {subject} - {name}\n"
    with open('attendance_log.txt', 'a') as f:
        f.write(entry)

    # Log to Excel
    date_str, time_str = timestamp.split(' ')
    excel_file = 'attendance_log.xlsx'
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=['Roll Number', 'Name', 'Date', 'Time', 'Subject'])

    # Extract roll number from name if possible (assuming name format "Name RollNumber")
    parts = name.rsplit(' ', 1)
    if len(parts) == 2:
        actual_name, roll_number = parts
        roll_number = str(roll_number)  # Ensure roll number is string to avoid float issues
    else:
        actual_name = name
        roll_number = ''

    new_row = {'Roll Number': roll_number, 'Name': actual_name, 'Date': date_str, 'Time': time_str, 'Subject': subject}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # Drop rows with any NaN values
    df = df.dropna(how='any')
    # Reorder columns before saving
    column_order = ['Roll Number', 'Name', 'Date', 'Subject', 'Time']
    df = df[column_order]
    df.to_excel(excel_file, index=False)

    return entry.strip()
import pickle

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
CAPTURED_IMAGES_DIR = 'captured_images'
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), '..', 'svm_classifier.pkl')

detector = MTCNN()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face(image, required_size=(160, 160)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_rgb[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return face

def get_embedding(face_pixels):
    face_pixels = np.expand_dims(face_pixels, axis=0)
    face_pixels = torch.tensor(face_pixels).permute(0, 3, 1, 2)  # Convert to NCHW format
    with torch.no_grad():
        embedding = facenet_model(face_pixels)
    return embedding[0].numpy()

import pickle
import sys
import json

# Redirect all print statements except the final JSON output to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

try:
    with open(CLASSIFIER_PATH, 'rb') as f:
        classifier, label_encoder = pickle.load(f)
    eprint("Classifier and label encoder loaded successfully.")
except Exception as e:
    eprint(f"Error loading classifier: {e}")
    classifier, label_encoder = None, None

def recognize_and_log(subject, image_filename=None):
    results = []
    if not os.path.exists(CAPTURED_IMAGES_DIR):
        eprint(f"No captured images directory found at {CAPTURED_IMAGES_DIR}")
        return []

    image_files = []
    if image_filename:
        image_files = [image_filename]
    else:
        image_files = [f for f in os.listdir(CAPTURED_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        image_path = os.path.join(CAPTURED_IMAGES_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            eprint(f"Failed to read image {image_path}")
            results.append({
                'image': filename,
                'predicted': None,
                'confidence': None,
                'logged': False,
                'message': 'Failed to read image'
            })
            continue
        face = extract_face(image)
        if face is None:
            eprint(f"No face detected in image {filename}")
            results.append({
                'image': filename,
                'predicted': None,
                'confidence': None,
                'logged': False,
                'message': 'No face detected'
            })
            continue
        embedding = get_embedding(face)
        embedding = embedding.reshape(1, -1)
        if classifier is None or label_encoder is None:
            eprint("Classifier or label encoder not loaded.")
            results.append({
                'image': filename,
                'predicted': None,
                'confidence': None,
                'logged': False,
                'message': 'Classifier not loaded'
            })
            continue
        pred_prob = classifier.predict_proba(embedding)[0]
        pred_class = np.argmax(pred_prob)
        confidence = pred_prob[pred_class]
        predicted_label = label_encoder.inverse_transform([pred_class])[0]
        eprint(f"Image: {filename}, Predicted: {predicted_label}, Confidence: {confidence}")
        if confidence > 0.5:
            entry = log_attendance(predicted_label, subject)
            results.append({
                'image': filename,
                'predicted': predicted_label,
                'confidence': round(float(confidence), 2),
                'logged': True,
                'message': f"Logged attendance: {entry}"
            })
        else:
            results.append({
                'image': filename,
                'predicted': predicted_label,
                'confidence': round(float(confidence), 2),
                'logged': False,
                'message': 'Low confidence, not logged'
            })
    return results

if __name__ == "__main__":
    subject_arg = sys.argv[1] if len(sys.argv) > 1 else 'Unknown'
    image_arg = sys.argv[2] if len(sys.argv) > 2 else None
    recognition_results = recognize_and_log(subject_arg, image_arg)
    print(json.dumps(recognition_results))
