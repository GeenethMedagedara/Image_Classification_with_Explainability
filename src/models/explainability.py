import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from keras.utils import array_to_img, img_to_array, load_img

class ExplainabilityStrategy:
    """Abstract class defining the explainability strategy."""
    def explain(self, model, image):
        raise NotImplementedError("Subclasses must implement 'explain' method.")

class GradCAM(ExplainabilityStrategy):
    """Concrete strategy for Grad-CAM visualization."""
    def __init__(self, layer_name="last_conv"):
        self.layer_name = layer_name

    def grad_cam(self, model, image, class_idx):
        """Computes Grad-CAM heatmap for a given image and class index."""
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        # Define model mapping inputs to activations & predictions
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(self.layer_name).output, model.output]
        )

        # Record gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image, training=False)
            loss = predictions[:, class_idx]  # Target class

        # Compute gradients
        grads = tape.gradient(loss, conv_output)
        if grads is None:
            raise ValueError("Gradient computation failed! Check model layers.")

        # Compute pooled gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps
        conv_output = conv_output[0]  # Remove batch dim
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= (np.max(heatmap) + 1e-10)  # Normalize

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """Overlays the heatmap on the original image."""
        heatmap = np.uint8(255 * heatmap)  # Convert to 0-255
        heatmap = np.expand_dims(heatmap, axis=-1)  # Add channel
        heatmap = np.repeat(heatmap, 3, axis=-1)  # Convert to RGB

        # Resize heatmap to match image size
        heatmap = tf.image.resize(heatmap, (image.shape[1], image.shape[2])).numpy()

        # Blend heatmap with image
        superimposed_image = np.uint8(image[0] * 255 * (1 - alpha) + heatmap * alpha)

        return superimposed_image

    def explain(self, model, image):
        """Main function to generate and display Grad-CAM results."""
        preds = model.predict(image)
        class_idx = np.argmax(preds)

        # Generate heatmap
        heatmap = self.grad_cam(model, image, class_idx)

        # Overlay heatmap
        overlayed_image = self.overlay_heatmap(heatmap, image)

        return heatmap, overlayed_image

class SaliencyMap(ExplainabilityStrategy):
    """Saliency Map explanation using gradients."""

    def explain(self, model, image):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.Variable(image)  # Convert to a variable for gradient computation

        with tf.GradientTape() as tape:
            tape.watch(image)
            predictions = model(image, training=False)
            class_idx = tf.argmax(predictions, axis=1).numpy()[0]  # Get predicted class
            loss = predictions[:, class_idx]  # Focus on target class

        gradients = tape.gradient(loss, image)  # Compute gradients
        gradients = tf.abs(gradients)  # Take absolute values (importance only)
        saliency = tf.reduce_max(gradients, axis=-1)  # Get max across color channels

        return saliency.numpy().squeeze()
    
class IntegratedGradients(ExplainabilityStrategy):
    """Integrated Gradients explanation technique."""

    def __init__(self, steps=20):
        self.steps = steps  # Number of steps for approximation

    def explain(self, model, image):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        baseline = tf.zeros_like(image)  # Black image as baseline
        diff = image - baseline  # Difference between image and baseline

        total_gradients = tf.zeros_like(image)  # Initialize accumulated gradients

        for i in range(self.steps + 1):
            # Interpolate the image along the path from baseline to input
            interpolated_image = baseline + (float(i) / self.steps) * diff

            with tf.GradientTape() as tape:
                tape.watch(interpolated_image)
                predictions = model(interpolated_image, training=False)
                class_idx = tf.argmax(predictions, axis=1).numpy()[0]  # Get predicted class
                loss = predictions[:, class_idx]

            # Compute gradients for the interpolated image
            gradients = tape.gradient(loss, interpolated_image)
            total_gradients += gradients  # Accumulate the gradients

        # Average the accumulated gradients
        avg_gradients = total_gradients / (self.steps + 1)

        # Normalize using ReLU and max scaling
        ig_map = np.maximum(avg_gradients.numpy().squeeze(), 0)
        ig_map /= (np.max(ig_map) + 1e-10)  # Normalize to [0,1]

        return ig_map
    
class GradCAMSuperimposed(ExplainabilityStrategy):
    """Grad-CAM explanation technique."""

    def __init__(self, last_conv_layer_name):
        self.last_conv_layer_name = last_conv_layer_name  # Specify last convolutional layer

    def explain(self, model, image):
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(self.last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image)
            class_idx = tf.argmax(preds[0])  # Get predicted class
            class_channel = preds[:, class_idx]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    @staticmethod
    def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        """Saves and displays the Grad-CAM heatmap superimposed on the image."""
        img = load_img(img_path)
        img = img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Apply colormap
        jet = plt.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Convert to image format
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # Superimpose the heatmap
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = array_to_img(superimposed_img)

        # Save and display
        superimposed_img.save(cam_path)
        return cam_path  # Return path to image