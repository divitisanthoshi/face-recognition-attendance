import os
import numpy as np
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2
from facenet_pytorch import InceptionResnetV1
import torch

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
EMBEDDINGS_FILE = 'embeddings.pkl'

detector = MTCNN()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face(image_path, required_size=(160, 160)):
    image = cv2.imread(image_path)
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

def extract_embeddings():
    embeddings = []
    labels = []
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            face = extract_face(image_path)
            if face is None:
                print(f"Face not detected in {image_path}")
                continue
            embedding = get_embedding(face)
            embeddings.append(embedding)
            labels.append(person_name)
            print(f"Processed {image_path}")
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump((embeddings, labels_enc, le), f)
    print(f"Saved embeddings and labels to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    extract_embeddings()
