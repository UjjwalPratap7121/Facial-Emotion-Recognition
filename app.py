import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import av

# ---------------------------------------------------------------
# CONFIG — change this if your model is somewhere else
# ---------------------------------------------------------------
MODEL_PATH = r"models/final_fer_model.keras"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


model = load_model()
face_cascade = load_face_detector()


def predict_face(gray_face_roi):
    """gray_face_roi: single-channel numpy array, any size."""
    face = cv2.resize(gray_face_roi, (48, 48)).astype("float32")
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    prediction = model.predict(face, verbose=0)[0]
    label_idx = int(np.argmax(prediction))
    return CLASS_NAMES[label_idx], float(prediction[label_idx]) * 100


st.title("🙂 Facial Emotion Recognition")
st.caption("CNN trained from scratch on FER2013-style data — 74% test accuracy")

tab1, tab2 = st.tabs(["📷 Upload a Photo", "🎥 Live Webcam"])

# ---------------------------------------------------------------
# TAB 1: PHOTO UPLOAD
# ---------------------------------------------------------------
with tab1:
    st.subheader("Upload a photo")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected in this image. Try another photo.")
            st.image(image, caption="Uploaded image", use_container_width=True)
        else:
            annotated = img_array.copy()
            results = []
            for (x, y, w, h) in faces:
                emotion, confidence = predict_face(gray[y:y + h, x:x + w])
                results.append((emotion, confidence))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(annotated, f"{emotion} ({confidence:.1f}%)",
                            (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 0), 2)

            st.image(annotated, caption="Prediction", use_container_width=True)
            for i, (emotion, confidence) in enumerate(results, 1):
                st.write(f"**Face {i}:** {emotion} ({confidence:.1f}% confidence)")

# ---------------------------------------------------------------

# TAB 2: CAMERA SNAPSHOT
# ---------------------------------------------------------------
with tab2:
    st.subheader("Take a photo")
    st.caption("Click below to open your camera and take a snapshot.")

    camera_photo = st.camera_input("Take a picture")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected. Try again with better lighting/angle.")
        else:
            annotated = img_array.copy()
            results = []
            for (x, y, w, h) in faces:
                emotion, confidence = predict_face(gray[y:y + h, x:x + w])
                results.append((emotion, confidence))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(annotated, f"{emotion} ({confidence:.1f}%)",
                            (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 0), 2)

            st.image(annotated, caption="Prediction", use_container_width=True)
            for i, (emotion, confidence) in enumerate(results, 1):
                st.write(f"**Face {i}:** {emotion} ({confidence:.1f}% confidence)")