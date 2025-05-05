# NN Final Project 
Title:
Seeing the Earth from Above: Teaching Computers to Understand Satellite Images

By: 
    Jack Beneigh: Bucknell University Junior Computer Science Engineering Major  
    Jonas Scott: Bucknell University Junior Computer Science Engineering Major  
    Sam Polakov: Colby College Junior Computer Science: Artificial Engineering Major  
    Anderson St. Clair: Washington University Junior Mathematics and Computer Science Major  

Plots of the Project:
    The various plots and graphs created for this project were set up to provide an understanding for how the model itself is working and to provide examples of the data we are using. Our model is used to understand various satellite images and classifying them into different geological categories. 

ImageExamples.png Code:
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
    #Call the function to display images    
    display_images(data_dir, categories)
        

ModelLossEp.png Code:

    #Model Accuracy Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


#Model Loss Plot

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


LearningRatesEp.png Code:

    #Learning Rate Graph
    plt.plot(history.history['learning_rate'])
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()


ConfusionMatricEp.png Code:

    #Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


PerClassAccuracyEp.png Code:

    #Per Class Accuracy Plot
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.bar(categories, class_accuracies)
    plt.title('Per-Class Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()