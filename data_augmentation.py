import os
import cv2
import numpy as np

def augment_image(image):
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Flip horizontally
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotate by 5 degrees
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)

    # Rotate by -5 degrees
    M = cv2.getRotationMatrix2D(center, -5, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)

    # Adjust brightness and contrast
    alpha = 1.1  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented_images.append(adjusted)

    # Add light Gaussian noise
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)

    return augmented_images

def augment_dataset(dataset_dir):
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read {image_path}")
                continue
            augmented_images = augment_image(image)
            base_name, ext = os.path.splitext(image_name)
            for i, aug_img in enumerate(augmented_images[1:], start=1):  # skip original
                new_name = f"{base_name}_aug{i}{ext}"
                new_path = os.path.join(person_dir, new_name)
                cv2.imwrite(new_path, aug_img)
                print(f"Saved augmented image {new_path}")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
    augment_dataset(dataset_path)
