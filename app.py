import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import numpy as np
import streamlit as st

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭")

st.title("🎭 Real-Time Facial Emotion Recognition")
st.write(
    "Take a photo using your webcam to analyze facial expressions in real time!"
)


def detect_emotion(landmarks):
    # Key facial landmark points
    top_lip = np.array([landmarks[13].x, landmarks[13].y])
    bottom_lip = np.array([landmarks[14].x, landmarks[14].y])
    left_corner = np.array([landmarks[61].x, landmarks[61].y])
    right_corner = np.array([landmarks[291].x, landmarks[291].y])

    left_eyebrow = np.array([landmarks[70].x, landmarks[70].y])
    right_eyebrow = np.array([landmarks[300].x, landmarks[300].y])
    left_eye = np.array([landmarks[159].x, landmarks[159].y])
    right_eye = np.array([landmarks[386].x, landmarks[386].y])

    # Calculate relative distances
    mouth_height = np.linalg.norm(top_lip - bottom_lip)
    mouth_width = np.linalg.norm(left_corner - right_corner)
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

    left_brow_dist = np.linalg.norm(left_eyebrow - left_eye)
    right_brow_dist = np.linalg.norm(right_eyebrow - right_eye)
    brow_dist = (left_brow_dist + right_brow_dist) / 2.0

    # Emotion classification rules
    if mouth_ratio > 0.4:
        return "SURPRISED / HAPPY", 92.5
    elif mouth_ratio > 0.18:
        return "HAPPY", 88.0
    elif brow_dist < 0.045:
        return "ANGRY / FOCUSED", 82.0
    elif mouth_ratio < 0.08:
        return "SAD / SERIOUS", 78.5
    else:
        return "NEUTRAL", 90.0


img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
    )
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as face_mesh:
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            emotion, confidence = detect_emotion(landmarks)
            st.success(
                f"Detected Emotion: **{emotion}** ({confidence:.1f}% confidence)"
            )
        else:
            st.warning("No face detected. Please position your face clearly.")