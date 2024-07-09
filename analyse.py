import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the input directory
input_directory = r'C:\Users\asus\Desktop\cognition\dataset\Training'  # Use raw string to handle backslashes

# Initialize lists to store image details
image_details = []

# Iterate through the classes (subdirectories)
for class_name in os.listdir(input_directory):
    class_path = os.path.join(input_directory, class_name)
    
    # Iterate through images in the class subdirectory
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if the image is loaded properly
        if image is not None:
            height, width, channels = image.shape
            image_type = image.dtype
            resolution = (height, width)
            
            # Append image details to the list
            image_details.append({
                'image_name': image_name,
                'class': class_name,
                'height': height,
                'width': width,
                'channels': channels,
                'type': image_type,
                'resolution': resolution,
                'path': image_path
            })

# Create a DataFrame from the image details
image_df = pd.DataFrame(image_details)

# Save the DataFrame to a CSV file
image_df.to_csv('image_details.csv', index=False)

# Detailed exploratory analysis
print("Number of images:", len(image_df))
print("Image classes:", image_df['class'].unique())
print("Average image resolution:", image_df[['height', 'width']].mean())
print("Image types:", image_df['type'].unique())

# Class-specific analysis
for class_name in image_df['class'].unique():
    class_df = image_df[image_df['class'] == class_name]
    print(f"\nClass: {class_name}")
    print(f"Number of images: {len(class_df)}")
    print(f"Average height: {class_df['height'].mean()}")
    print(f"Average width: {class_df['width'].mean()}")
    print(f"Height variance: {np.var(class_df['height'])}")
    print(f"Width variance: {np.var(class_df['width'])}")

    # Show a sample image from each class
    sample_image_path = class_df.iloc[0]['path']
    image = cv2.imread(sample_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Sample from {class_name}")
    plt.axis('off')
    plt.show()
