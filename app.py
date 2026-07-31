import cv2
import numpy as np
import streamlit as st

# Safe import for fer detector
try:
    from fer import FER
except ImportError:
    from fer.fer import FER

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭")

st.title("🎭 Real-Time Facial Emotion Recognition")
st.write(
    "Take a photo using your webcam to analyze facial expressions in real time!"
)

# Initialize detector
@st.cache_resource
def load_detector():
    return FER(mtcnn=False)

detector = load_detector()

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
    )

    with st.spinner("Analyzing emotion..."):
        # Detect top emotion
        result = detector.top_emotion(cv2_img)

        if result and result[0] is not None:
            emotion, score = result
            st.success(
                f"Detected Emotion: **{emotion.upper()}** ({score*100:.1f}% confidence)"
            )
        else:
            st.warning(
                "No face detected. Please make sure your face is clearly visible and well-lit."
            )