import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def build_model_and_load_weights(weights_path):
    """Rebuilds the architecture and loads weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

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
    
    model.load_weights(weights_path)
    return model

def main():
    weights_path = 'micro_expression.h5'
    
    print("Loading model weights...")
    model = build_model_and_load_weights(weights_path)
    print("Model loaded. Initializing webcam...")

    # Load OpenCV's built-in face detector
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 0 is usually the default built-in webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        
        # Convert frame to grayscale for face detection and model input
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region of interest (ROI)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Resize to match model input shape (48x48)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                # Normalize and format for the model
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)      # Add batch dimension
                roi = np.expand_dims(roi, axis=-1)     # Add channel dimension -> (1, 48, 48, 1)

                # Predict emotion
                prediction = model.predict(roi, verbose=0)
                label_index = np.argmax(prediction)
                label = CLASS_NAMES[label_index]
                confidence = round(prediction[0][label_index] * 100, 2)
                
                # Display text
                text = f"{label} ({confidence}%)"
                label_position = (x, y - 10)
                cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Face', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Live Emotion Detection (Press "q" to quit)', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()