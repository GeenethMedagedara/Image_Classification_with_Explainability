"""
Handles prediction strategies for image classification models.
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class PredictionStrategy:
    """Abstract class defining the prediction strategy."""
    def predict(self, model, image_path, class_labels):
        raise NotImplementedError("Subclasses must implement 'predict' method.")

class ImageClassifier(PredictionStrategy):
    """Concrete strategy for image classification."""
    def predict(self, model, image_path, class_labels):
        # Load and display the image
        img = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        # Resize the image to match model input shape
        resized_img = tf.image.resize(img, (32, 32))
        plt.imshow(resized_img.numpy().astype(int))
        plt.show()

        # Expand dimensions to match batch format
        input_tensor = np.expand_dims(resized_img / 255.0, axis=0)

        # Make prediction
        prediction = model.predict(input_tensor)

        # Get predicted class
        predicted_class = np.argmax(prediction)
        class_name = class_labels[predicted_class]

        print(f"Predicted Class: {class_name}")
        return class_name, prediction

class PredictionContext:
    """Context class to execute different prediction strategies."""
    def __init__(self, strategy: PredictionStrategy):
        self.strategy = strategy

    def execute_prediction(self, model, image_path, class_labels):
        return self.strategy.predict(model, image_path, class_labels)
