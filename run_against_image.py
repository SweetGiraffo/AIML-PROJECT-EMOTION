import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the exact class names used during training
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def build_model_and_load_weights(weights_path):
    """Rebuilds the architecture and loads legacy .h5 weights to bypass Keras 3 config errors."""
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Rebuild the exact architecture
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(7, activation='softmax')
    ])
    
    # Load only the weights, bypassing the broken JSON configuration
    model.load_weights(weights_path)
    print(f"Weights loaded successfully from {weights_path}")
    
    return model

def detect_emotion(model, image_path):
    """Runs inference on a single image using the loaded model."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None

    # Load and preprocess the image to match training parameters
    img = load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    # Display results
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Emotion: {predicted_class} ({confidence}%)')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return predicted_class, confidence

if __name__ == "__main__":
    # 1. Path to your legacy .h5 file
    WEIGHTS_FILE = 'micro_expression.h5'
    
    # 2. Rebuild architecture and load weights
    model = build_model_and_load_weights(WEIGHTS_FILE)
    
    # 3. Test the model on an image
    # Replace 'test.jpg' with the actual path to your test image
    test_image_path = 'test.jpg' 
    
    if os.path.exists(test_image_path):
        emotion, conf = detect_emotion(model, test_image_path)
        print(f"Result: {emotion} at {conf}% confidence.")
    else:
        print(f"Place a test image named '{test_image_path}' in the directory to run a prediction.")