import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

model = YOLO(r"C:\Users\jessl\repos\yolo-handcart-detection\src\data_pipeline\runs\detect\v1_best_diff_trolly-2\weights\best.pt")

st.title("🚗 Trolley / Handcart Detection")

mode = st.radio("Select Mode", ["Image Upload", "Real-Time Camera"])

# -------------------------
# 1. IMAGE MODE
# -------------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image")

    if uploaded_file:
        image = Image.open(uploaded_file)
        image = np.array(image)

        results = model(image)
        annotated = results[0].plot()

        st.image(annotated, caption="Detection Result")

# -------------------------
# 2. REAL-TIME CAMERA MODE
# -------------------------
elif mode == "Real-Time Camera":
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        FRAME_WINDOW.image(annotated, channels="BGR")

    camera.release()