# Facial Emotion Recognition (FER) via CNN

## Project Team
* **Vaibhav Choudhary** - Roll No: 25MA60R29
* **Keshav** - Roll No: 25MA60R10
* **Mrinmoy Mondal** - Roll No: 25MA60R09
* **Rani** - Roll No: 25MA60R20
* **Urmila** - Roll No: 25MA60R23

---

## Overview
This project implements a Convolutional Neural Network (CNN) to classify human facial emotions into seven distinct categories. The pipeline covers data ingestion, augmentation, model training, evaluation, static image inference, and real-time webcam detection.

## Dependencies
The code requires the following Python libraries:
* `tensorflow` / `keras`
* `opencv-python` (`cv2`)
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `kagglehub`

## Dataset
The project uses the **FER2013** dataset, downloaded dynamically via `kagglehub` (`msambare/fer2013`). 
* **Classes (7):** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
* **Format:** 48x48 pixel grayscale images.
* **Volume:** 28,709 training images, 7,178 validation/test images.

## Model Architecture
A sequential CNN model built with Keras, comprising approximately 2.47 million parameters:
1. **Input:** 48x48x1 Grayscale images.
2. **Convolutional Blocks:** Three sequential blocks containing `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout` layers. Filter sizes progress from 64 -> 128 -> 256.
3. **Fully Connected Layers:** `Flatten` followed by a 512-neuron `Dense` layer, `BatchNormalization`, and `Dropout`.
4. **Output:** A 7-neuron `Dense` layer with a softmax activation function.

**Compilation Parameters:**
* **Optimizer:** Adam (Learning Rate: 0.001)
* **Loss Function:** Categorical Crossentropy

## Workflow / Usage

### 1. Training Setup
Data is fed into the model using `ImageDataGenerator`. The training set applies data augmentation (rotation, zoom, horizontal flip, rescaling). The validation set only applies rescaling. 
* To train the model, execute the `model.fit` cell. 
* The script trains for 100 epochs by default.

### 2. Evaluation
After training, the script generates:
* Matplotlib charts for Training/Validation Accuracy and Loss.
* A Classification Report using `sklearn.metrics`.
* A visual Confusion Matrix plotted with Seaborn to map predicted versus actual classifications.

### 3. Model Saving
The trained weights and architecture are serialized and saved locally as:
`micro_expression.h5`

### 4. Inference on Static Images
The `detect_emotion(image_path)` function accepts a local file path, preprocesses the image to match the 48x48 input shape, runs it through the saved model, and outputs the predicted emotion and confidence percentage. 

*Note: Update the hardcoded `image_path` paths (e.g., `E:\Muqadas\...`) in the script to match your local file structure before running static inference.*

### 5. Live Detection (Webcam)
The final section of the code initializes a live OpenCV video capture.
* Uses `haarcascade_frontalface_default.xml` to detect faces in the video stream.
* Crops and resizes detected faces to 48x48.
* Passes the cropped face to `micro_expression.h5` for real-time inference.
* Draws a bounding box and overlays the predicted emotion on the video feed.
* **To exit:** Press the `q` key.
