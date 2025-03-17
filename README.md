# Image Classification with Explainability

This project is about classifying images and also this enhances model interpretability by generating **Grad-CAM and Integrated Gradients visualizations** for image classification models. It helps users understand *why* a model makes certain predictions—making AI more transparent and trustworthy.

---

## Features

- **AI Explainability:** Uses Grad-CAM & Integrated Gradients to highlight key regions.
- **End-to-End Pipeline:** Loads images, runs inference, and generates visual explanations. 
- **Flask Backend:** Serves predictions via a REST API.
- **React Frontend:** Provides an interactive UI for users.

## Tech Stack

- **Machine Learning:** Tensorflow, Keras, MLflow, Grad-CAM, Integrated Gradients, OpenCV
- **Backend:** Flask
- **Frontend:** React, Tailwind CSS  
  

## Installation

1. Clone the Repository
```
git clone https://github.com/GeenethMedagedara/Image_Classification_with_Explainability.git
cd explainable-ai-classifier
```

2. Install Dependencies
```
pip install -r requirements.txt

# Go to frontend directory
cd frontend
npm install
```

2. To run mlflow
```
mlflow ui
```

3. To train the model

Run the two notebooks data_exploration.ipynb and model_training.ipynb respectively 

img

4. Run the App
```
# To run react app
cd frontend 
npm run dev

# To run flask backend
cd backend
flask run --host=0.0.0.0 --port=4000
```

5. Access the frontend at
```
http://localhost:8080
```

## Model Explainability in Action

<table>
  <tr>
    <th>Original Image</th>
    <th>Grad-CAM Visualization</th>
    <th>Original Superimposed image with Grad-CAM</th>
    <th>Integrated Gradients</th>
  </tr>
  <tr>
    <td><img src="docs/original_image.jpg" width="200"></td>
    <td><img src="docs/gradcam.jpg" width="200"></td>
    <td><img src="docs/original_superimposed_image.jpg" width="200"></td>
    <td><img src="docs/integrated_gradients.jpg" width="200"></td>
  </tr>
</table>

## How This Works
I created this Convolutional Neural Network (CNN) using tensorflow functional API for easier Grad-Cam integration.
A Convolutional Neural Network (CNN) is designed to recognize patterns in images through a series of learnable filters.
- **Convolution Layers:** Extract key features (edges, textures, shapes) by sliding filters over the image.
- **Activation Function (ReLU):** Introduces non-linearity, allowing the network to learn complex patterns.
- **Pooling Layers (MaxPooling):** Reduce dimensionality while retaining important features, making computation more efficient.

Deeper the network goes, it detects patterns and then later layers combine these patterns and recognize objects

- **Dense layers:** Which are fully connected gets the flattened and extracted features
- **Softmax function:** assigns probabilities to each class, selecting the most likely one.

The CNN compares predictions with the actual labels using a loss function (e.g., Cross-Entropy).
Backpropagation adjusts filter weights using Gradient Descent (e.g., Adam, SGD) to minimize errors over time.

The Grad-Cam works by computing gradients of the predicted class score(final layer) **w.r.t** the feature maps of the last convolutional layer.
## Why This is Built

I built this project to get an understanding of Convolutional Neural Network architecture, model building, and how these models perform predictions. This gives me a basic understanding of computer vision and explainable AI.

## Screenshots


