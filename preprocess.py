import os

import cv2
import numpy as np
from tqdm import tqdm


def preprocess_image(image_path, output_path, size=(128, 128)):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return

    # Resize image
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Normalize pixel values
    img_normalized = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX)

    # Save preprocessed image
    cv2.imwrite(output_path, img_normalized)

def preprocess_dataset(input_dir, output_dir, size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = ['Empty', 'High', 'Low', 'Medium']  # Adjust these as per your dataset
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        for img_name in tqdm(os.listdir(input_class_dir), desc=f'Processing {class_name}'):
            input_img_path = os.path.join(input_class_dir, img_name)
            output_img_path = os.path.join(output_class_dir, img_name)
            preprocess_image(input_img_path, output_img_path, size)

if __name__ == '__main__':
    input_dir = r'C:\Users\asus\Desktop\cognition\dataset\Training'
    output_dir = r'C:\Users\asus\Desktop\cognition\preprocessed-dataset'
    preprocess_dataset(input_dir, output_dir)
