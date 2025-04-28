import cv2
import os
import time

def capture_image(save_path='captured_images', delay=0.5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    try:
        print(f"Starting camera. Capturing image in {delay} seconds...")
        time.sleep(delay)
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(save_path, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image saved to {filename}")
            return filename
        else:
            print("Error: Could not read frame from camera.")
            return None
    except Exception as e:
        print(f"Exception during image capture: {e}")
        return None
    finally:
        cap.release()

if __name__ == "__main__":
    capture_image()
