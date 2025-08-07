import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
try:
    model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels mapping
class_labels = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Non-proliferative Diabetic Retinopathy',
    2: 'Moderate Non-proliferative Diabetic Retinopathy',
    3: 'Severe Non-proliferative Diabetic Retinopathy',
    4: 'Proliferative Diabetic Retinopathy'
}

def predict_image(image_path):
    # Load and preprocess the image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to read the image at {image_path}")
            return None
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class}")
        return predicted_class
    except Exception as e:
        print(f"Error during image processing: {e}")
        return None
