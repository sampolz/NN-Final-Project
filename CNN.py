import os
import cv2
import tensorflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, Xception, InceptionResNetV2, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


data_dir = 'data' # Data File

categories  = ['cloudy', 'desert', 'green_area', 'water']
data = []
data_labels = []

# Loop through each category and collect image paths
for categorie in categories :
    folder_path = os.path.join(data_dir, categorie)
    image_paths = os.listdir(folder_path)
    data.extend([os.path.join(folder_path, image_path) for image_path in image_paths])
    data_labels.extend([categorie] * len(image_paths))

# Create a DataFrame with the image paths and labels
data = pd.DataFrame({'Image_Path': data, 'labels': data_labels})
print(data) # Display the DataFrame

# Function to display images
def display_images(data_dir, categories, num_images=10):
    # Set up the plot grid (2 rows and 5 columns, change if needed)
    fig, axes = plt.subplots(len(categories), num_images, figsize=(20, len(categories)*2))
    
    # Loop through each category
    for i, category in enumerate(categories):
        # Get the paths of images in this category
        folder_path = os.path.join(data_dir, category)
        image_paths = os.listdir(folder_path)
        
        # Randomly select num_images images
        selected_images = random.sample(image_paths, num_images)

        for j, image_path in enumerate(selected_images):
            # Construct the full image path
            img_path = os.path.join(folder_path, image_path)
            
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display in matplotlib
            
            # Plot the image in the grid
            axes[i, j].imshow(img)
            axes[i, j].axis('off')  # Turn off axis
            axes[i, j].set_title(category)  # Label with category name
    
    plt.tight_layout()  # Adjust spacing for better readability
    plt.show()

# Call the function to display images
display_images(data_dir, categories)