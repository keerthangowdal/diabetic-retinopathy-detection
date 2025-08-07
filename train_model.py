import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json

# ----------------------------
# 1. Utility Functions
# ----------------------------

def load_images_and_labels(image_folder, label_file, image_size=(224, 224)):
    """Load images and corresponding labels."""
    images, labels = [], []
    for filename in os.listdir(image_folder):
        img_filename = os.path.splitext(filename)[0]
        img_path = os.path.join(image_folder, filename)
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
        
        # Match image to its label
        label = label_file[label_file['id_code'] == img_filename]
        if label.empty:
            print(f"Warning: No label found for {filename}")
        else:
            labels.append(label['diagnosis'].values[0])
    
    return np.array(images), np.array(labels)

def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ----------------------------
# 2. Load and Preprocess Data
# ----------------------------

# Paths to data
image_folder_train = 'data/train_images'
image_folder_valid = 'data/valid_images'
label_file_train = 'data/train.csv'
label_file_valid = 'data/valid.csv'

# Load CSV files
labels_train = pd.read_csv(label_file_train)
labels_valid = pd.read_csv(label_file_valid)

# Load images and labels
images_train, labels_train = load_images_and_labels(image_folder_train, labels_train)
images_valid, labels_valid = load_images_and_labels(image_folder_valid, labels_valid)

# Normalize images
images_train = images_train / 255.0
images_valid = images_valid / 255.0

# Encode labels
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)
labels_valid_encoded = label_encoder.transform(labels_valid)

# One-hot encoding
labels_train_onehot = to_categorical(labels_train_encoded, num_classes=5)
labels_valid_onehot = to_categorical(labels_valid_encoded, num_classes=5)

# ----------------------------
# 3. Data Augmentation
# ----------------------------

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(images_train)

# ----------------------------
# 4. Model Definition
# ----------------------------

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(5, activation='softmax')  # 5 classes
    ])
    return model

# Compile the model
model = create_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# ----------------------------
# 5. Training the Model
# ----------------------------

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    datagen.flow(images_train, labels_train_onehot, batch_size=32),
    validation_data=(images_valid, labels_valid_onehot),
    epochs=10,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Save final model
model.save('final_model.h5')

# ----------------------------
# 6. Evaluation and Visualization
# ----------------------------

# Evaluate the model
val_loss, val_accuracy = model.evaluate(images_valid, labels_valid_onehot)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the training history
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# Plot training history
plot_training_history(history)
