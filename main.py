from __future__ import annotations  # <-- MUST BE THE FIRST LINE
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppresses TensorFlow warnings

import cv2
from deepface import DeepFace

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam (0 is default laptop camera)
cap = cv2.VideoCapture(0)

print("Starting Webcam... Press 'q' on the camera window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam.")
        break

    # Convert frame to grayscale for faster face tracking
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in live video
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face area
        face_roi = frame[y:y + h, x:x + w]

        try:
            # Analyze emotion
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # DeepFace returns a list of dicts or a dict depending on version
            if isinstance(results, list):
                dominant_emotion = results[0]['dominant_emotion']
            else:
                dominant_emotion = results['dominant_emotion']

            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show emotion text above the face box
            cv2.putText(frame, f"Emotion: {dominant_emotion.upper()}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception as e:
            pass

    # Display the live video feed window
    cv2.imshow('Facial Emotion Detector', frame)

    # Press 'q' key to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()