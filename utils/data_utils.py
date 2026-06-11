import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import os
import shutil

# Constants 
TARGET_SIZE = (224, 224) # Standard input size (from common practice/Class 15 style)
BATCH_SIZE = 32
SEED = 42
TEST_SIZE_RATIO = 0.10      # 10% for final test set (Matches Class 15 style)
VALIDATION_SIZE_RATIO = 0.10 # 10% for validation set

#  1. Data Splitting Function 

def perform_stratified_split(metadata_df, image_root, data_target_dir='data'):
    """
    Splits the data into Train, Validation, and Test sets (e.g., 80/10/10) 
    using stratification on the 'family' column, then organizes images into directories.
    """
    X = metadata_df['file_path']
    y = metadata_df['family']

    # Step 1: Separate the final Test set (10% stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, random_state=SEED, stratify=y
    )
    
    # Calculate Validation ratio relative to the remaining data
    val_ratio_from_temp = VALIDATION_SIZE_RATIO / (1.0 - TEST_SIZE_RATIO) 
    
    # Step 2: Separate Training and Validation sets (10% of total split from temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_from_temp, random_state=SEED, stratify=y_temp
    )

    print(f"Data Split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    # --- Organize Files into Directories ---
    
    split_data = {
        'train': pd.concat([X_train, y_train], axis=1).rename_axis('file_path'),
        'validation': pd.concat([X_val, y_val], axis=1).rename_axis('file_path'),
        'test': pd.concat([X_test, y_test], axis=1).rename_axis('file_path')
    }

    if os.path.exists(data_target_dir):
        # Optional: Clean up existing data directory before creating new split
        # shutil.rmtree(data_target_dir)
        pass 
    os.makedirs(data_target_dir, exist_ok=True)

    for split_name, split_df in split_data.items():
        print(f"Organizing {split_name} set...")
        for file_path_rel, family in split_df[['file_path', 'family']].values:
            
            # Create target structure: data/split/family/image.jpg
            target_family_dir = os.path.join(data_target_dir, split_name, family)
            os.makedirs(target_family_dir, exist_ok=True)
            
            # Define paths
            source_path = os.path.join(image_root, file_path_rel)
            destination_path = os.path.join(target_family_dir, os.path.basename(file_path_rel))
            
            # Copy file (safer than moving)
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
            # Note: You should check for broken paths in the EDA notebook first.

    print("Data directory structure successfully created/updated.")
    return os.path.abspath(data_target_dir) # Return the base path for generators

# 2. Resizing the images

def resize_and_create_arrays(directory, X, y):
    for idx, folder in enumerate(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path) and folder in source_folders:
            label = source_folders.index(folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                with Image.open(file_path) as img:
                    # Resize the image to 50x50
                    img_resized = img.resize((50, 50))
                    # Convert the image to numpy array
                    img_array = np.array(img_resized)
                    # Append the image array to X
                    X.append(img_array)
                    # Append the corresponding label to y
                    y.append(label)

# 3. Data Generator Class (Custom CNN) 

class DataGeneratorUtils:
    """ Utility class to create Keras generators, primarily for Custom CNN (0-1 normalization). """

    def __init__(self, data_root):
        self.data_root = data_root

    def _custom_preprocess(self, image):
        """ Handles Grayscale to RGB conversion. Rescaling (0-1) is handled by ImageDataGenerator(rescale). """
        if tf.shape(image)[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        return image

    def create_generators(self, set_type):
        """ Creates an image generator using 0-1 scaling and custom augmentation/preprocessing. """
        data_dir = os.path.join(self.data_root, set_type)
        
        if set_type == 'train':
            # Data Augmentation for Training set (Similar to Class 15, but customized)
            datagen = ImageDataGenerator(
                rescale=1./255, # 0-1 Normalization (similar to Class 15)
                preprocessing_function=self._custom_preprocess, # Handles grayscale to RGB
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            # Only 0-1 Normalization for Validation and Test sets
            datagen = ImageDataGenerator(
                rescale=1./255, # 0-1 Normalization (similar to Class 15)
                preprocessing_function=self._custom_preprocess
            )

        # Flow from Directory (Image of Keras flow_from_directory structure)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=(set_type == 'train'), 
            seed=SEED
        )
        return generator

    def calculate_class_weights(self, train_generator):
        """ Calculates class weights for imbalanced data. """
        labels = train_generator.classes
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights_dict = dict(enumerate(class_weights_array))
        print(f"Class weights calculated for {len(class_weights_dict)} classes.")
        return class_weights_dict

