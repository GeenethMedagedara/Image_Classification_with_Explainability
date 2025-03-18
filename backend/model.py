"""
Handles model loading and prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "../notebooks/saved_model/final_model.h5"
model = load_model(MODEL_PATH)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # Modify based on your dataset

def preprocess_image(image_path, target_size=(32, 32)):
    """Loads and preprocesses an image for model inference."""
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = image.resize(target_size)  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print("Preprocessed image shape:", image.shape)
    return image

def predict(image_path):
    """Runs model prediction on an input image."""
    image = preprocess_image(image_path)
    predictions = model.predict(image)[0]
    predicted_class_idx = np.argmax(predictions)
    return CLASS_NAMES[predicted_class_idx], predictions.tolist()
