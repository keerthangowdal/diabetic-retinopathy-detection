import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

def load_labels(label_file):
    return pd.read_csv(label_file)

def load_images_and_labels(image_folder, label_file, img_size=(224, 224)):
    images, labels = [], []
    for filename in tqdm(os.listdir(image_folder), desc=f"Loading from {image_folder}"):
        img_filename = os.path.splitext(filename)[0]
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        images.append(img)

        label = label_file[label_file['id_code'] == img_filename]
        if label.empty:
            print(f"Warning: No label found for {filename}")
        else:
            labels.append(label['diagnosis'].values[0])
    return np.array(images), np.array(labels)

def preprocess_data(images, labels, test_size=0.2, num_classes=5):
    X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=test_size, random_state=42)
    X_train, X_valid = X_train / 255.0, X_valid / 255.0

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    
    y_train_onehot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_valid_onehot = to_categorical(y_valid_encoded, num_classes=num_classes)

    return X_train, X_valid, y_train_onehot, y_valid_onehot

# Paths
image_folder_train = 'data/train_images'
image_folder_valid = 'data/valid_images'
label_file_train = 'data/train.csv'
label_file_valid = 'data/valid.csv'

# Load and preprocess data
labels_train = load_labels(label_file_train)
labels_valid = load_labels(label_file_valid)

train_images, train_labels = load_images_and_labels(image_folder_train, labels_train)
valid_images, valid_labels = load_images_and_labels(image_folder_valid, labels_valid)

X_train, X_valid, y_train, y_valid = preprocess_data(train_images, train_labels)
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_valid.shape}")
