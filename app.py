import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np

st.title("🎭 Real-Time Facial Emotion Detection")
st.write("Take a picture or use your webcam to test real-time facial emotion recognition!")

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Analyze emotion
    try:
        results = DeepFace.analyze(cv2_img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = results[0]['dominant_emotion']
        st.success(f"Detected Emotion: **{dominant_emotion.upper()}**")
    except Exception as e:
        st.error("Could not analyze the image. Make sure your face is clearly visible.")