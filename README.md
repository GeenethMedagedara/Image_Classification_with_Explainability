# Image Classification with Explainability

![Screenshot 2025-03-17 123740](https://github.com/user-attachments/assets/f09607dc-6d4e-4963-95bb-223d87942820)

---

This project is about classifying images and also this enhances model interpretability by generating **Grad-CAM and Integrated Gradients visualizations** for image classification models. It helps users understand *why* a model makes certain predictions—making AI more transparent and trustworthy.

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

![Screenshot 2025-03-17 105437](https://github.com/user-attachments/assets/d2fc06eb-6a32-4e96-92b1-05f34004b011)

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
    <td><img src="https://github.com/user-attachments/assets/7ff969df-baea-4a95-8f8b-f0c9cf5fb9f5
" width="300" height="200"></td>
    <td><img src="![output2](https://github.com/user-attachments/assets/6c5b268e-77b4-4cee-a9f8-e2384e2126e8)
" width="200"></td>
    <td><img src="![output](https://github.com/user-attachments/assets/18c6de0d-c842-4d9c-ba11-f100d4bff5d9)
" width="200"></td>
    <td><img src="![output](https://github.com/user-attachments/assets/d63236e7-e08d-4766-a259-bb96142f67b7)
" width="200"></td>
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


