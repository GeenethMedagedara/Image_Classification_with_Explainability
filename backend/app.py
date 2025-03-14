from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
import base64

from model import predict
from explainability_flask import grad_cam, integrated_gradients, superimposed

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": [
            "http://host.docker.internal:8080",
            "http://localhost:8080",
            "http://backend:4000"  # If using Docker Compose
        ],
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    }
})


# Load the model
model = tf.keras.models.load_model("../notebooks/saved_model/final_model.h5")

def encode_image(image_path):
    """Convert an image file to Base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

@app.route("/predict", methods=["POST"])
def predict_image():
    """Endpoint to classify an image and provide explainability."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    predicted_class, probabilities = predict(file_path)
    
    # Grad-CAM explainability
    gradcam_path = grad_cam(file_path, model)

    # Integrated Gradients
    class_idx = np.argmax(probabilities)
    ig_path = integrated_gradients(file_path, model, class_idx)
    
    super_path = superimposed(file_path, model)
    
    # Convert images to Base64
    gradcam_base64 = encode_image(gradcam_path)
    ig_base64 = encode_image(ig_path)
    superimposed_base64 = encode_image(super_path)

    return jsonify({
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "gradcam_image": gradcam_base64,
        "integrated_gradients_image": ig_base64,
        "superimposed_image": superimposed_base64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
