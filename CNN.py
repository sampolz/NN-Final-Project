import os
import cv2
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
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
def display_images(data_dir, categories, num_images=5):
    # Set up the plot grid (2 rows and 5 columns, change if needed)
    fig, axes = plt.subplots(len(categories), num_images, figsize=(10, len(categories)*2))
    
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
    

# Set image and batch size for training
img_size = (75, 75)
batch_size = 32

# Map the category labels to a numerical index
label_map = {label: idx for idx, label in enumerate(categories)}
data['label_idx'] = data['labels'].map(label_map)

# Split into training and validation set with 80/20 split
train_df, val_df = train_test_split(data, test_size=0.2, stratify=data['label_idx'], random_state=42)

# Function to load and preprocess a single image from the file path
def load_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img, tf.one_hot(label, depth=len(categories))

# # tf dataset from dataframe
# def create_dataset(df):
#     paths = df['Image_Path'].values
#     labels = df['label_idx'].values
#     dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
#     dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
#     return dataset

# # Prepare training data by random shuffling, batching into groups of 32, and prefetching the batches for performacne
# train_ds = create_dataset(train_df).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
# # Create and prepare validation dataset with no shuffling, to preserve the evaluation considency. 
# val_ds = create_dataset(val_df).batch(32).prefetch(tf.data.AUTOTUNE)



# # Model Architecture
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(len(categories), activation='softmax')  # 4 classes
# ])

# # Compile model
# model.compile(
#     optimizer=Adam(),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# class LearningRateLogger(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         lr = self.model.optimizer.learning_rate
#         # Check if it's a schedule (e.g. with ReduceLROnPlateau) or a float
#         if hasattr(lr, 'numpy'):
#             logs['learning_rate'] = lr.numpy()
#         else:
#             logs['learning_rate'] = tf.keras.backend.eval(lr)


# # Train
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=15,
#     callbacks=[
#         EarlyStopping(patience=3, restore_best_weights=True),
#         ReduceLROnPlateau(patience=2),
#         LearningRateLogger()
#     ]
# )


# # Save the trained model
# model.save('trained_model.keras')

# # Save the training history to CSV
# history_df = pd.DataFrame(history.history)
# history_df.to_csv('training_history.csv', index=False)


# # Plot training & validation accuracy values
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()




