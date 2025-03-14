import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import os

from model import preprocess_image
from tensorflow.keras.models import Model
from keras.utils import array_to_img, img_to_array, load_img

# Grad-CAM Explainability
def grad_cam(image_path, model, last_conv_layer_name="last_conv"):
    """Computes Grad-CAM heatmap for a given image."""
    image = preprocess_image(image_path)

    # Create a model that maps inputs to activations & predictions
    grad_model = Model(inputs=model.input, 
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_output, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Apply Grad-CAM
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Save heatmap
    heatmap_path = f"./static/gradcam_{os.path.basename(image_path)}"
    plt.matshow(heatmap)
    plt.axis("off")
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    return heatmap_path

# Integrated Gradients Explainability
def integrated_gradients(image_path, model, class_idx, steps=50):
    """Computes Integrated Gradients for a given image."""
    image = preprocess_image(image_path)  # Expected shape: (1, 32, 32, 3)
    baseline = tf.zeros_like(image)

    # Compute path integral
    interpolated_images = [(baseline + (float(i) / steps) * (image - baseline)) for i in range(steps + 1)]
    interpolated_images = tf.convert_to_tensor(interpolated_images)

    # Fix shape before passing to model
    interpolated_images = tf.squeeze(interpolated_images, axis=1)  # Shape becomes (51, 32, 32, 3)

    print(f"Fixed interpolated_images shape: {interpolated_images.shape}")  # Debugging

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images, training=False)
        target_class_predictions = predictions[:, class_idx]

    # Compute gradients & integrate
    grads = tape.gradient(target_class_predictions, interpolated_images)
    avg_gradients = tf.reduce_mean(grads, axis=0)

    # Normalize and visualize
    ig_map = tf.reduce_mean(avg_gradients, axis=-1).numpy().squeeze()
    ig_map = np.maximum(ig_map, 0)
    ig_map /= np.max(ig_map) + 1e-10

    # Save IG heatmap
    ig_path = f"./static/integrated_gradients_{os.path.basename(image_path)}"
    plt.imshow(ig_map, cmap="hot")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(ig_path, bbox_inches="tight")
    plt.close()
    return ig_path




def superimposed(image_path, model, last_conv_layer_name="last_conv"):
    """Computes Superimposed Grad-CAM heatmap for a given image."""
    image = preprocess_image(image_path)
        
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
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
    heatmap =  heatmap.numpy()
        
    """Saves and displays the Grad-CAM heatmap superimposed on the image."""
    img = load_img(image_path)
    img = img_to_array(img)
    alpha=0.4

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
    superimposed_img = np.array(superimposed_img).astype("uint8")
    
    
    # Plot with Matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(superimposed_img)
    ax.axis("off")
    
    # Save Matplotlib figure
    cam_path = f"./static/superimposed_{os.path.basename(image_path)}"
    plt.savefig(cam_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return cam_path   # Return path to image