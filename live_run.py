import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load model once
model = load_model('micro_expression.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

cap = cv2.VideoCapture(0)
# Ensure the cascade file exists
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Failed to load haarcascade. Check your OpenCV installation.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

        # Faster inference for real-time loops
        prediction = model(roi_reshaped, training=False)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # max(y-10, 10) prevents text from going off-top
        cv2.putText(frame, emotion, (x, max(y-10, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Live Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
os.system("pause")
cap.release()
cv2.destroyAllWindows()
os.system("pause")