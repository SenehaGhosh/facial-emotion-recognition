import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭")

st.title("🎭 Real-Time Facial Emotion Recognition")
st.write(
    "Take a photo using your webcam to analyze facial expressions in real time!"
)

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV image format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
    )

    with st.spinner("Analyzing emotion..."):
        try:
            # Analyze emotions using DeepFace
            predictions = DeepFace.analyze(
                img_path=cv2_img, actions=["emotion"], enforce_detection=False
            )

            if predictions:
                dominant_emotion = predictions[0]["dominant_emotion"]
                scores = predictions[0]["emotion"]
                confidence = scores[dominant_emotion]

                st.success(
                    f"Detected Emotion: **{dominant_emotion.upper()}** ({confidence:.1f}% confidence)"
                )
            else:
                st.warning("No face detected. Please try again.")

        except Exception as e:
            st.error(
                "Could not process image. Ensure your face is clearly visible and well lit."
            )