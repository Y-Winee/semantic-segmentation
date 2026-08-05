import cv2
import streamlit as st
import numpy as np
from PIL import Image

from model import load_model
from predict import predict
from visualization import colorize, overlay

model = load_model()

uploaded = st.file_uploader(
    "Upload Image",
    ["jpg","jpeg","png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    original_h, original_w = image_np.shape[:2]

    mask = predict(model, image_np)
    color_mask = colorize(mask)
    color_mask = cv2.resize(
        color_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )
     
    st.write("Original:", image_np.shape)
    st.write("Mask:", color_mask.shape)

    col1, col2 = st.columns(2)

    overlay_img = overlay(image_np, color_mask)

    with col1:
        st.subheader("Original")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Overlay")
        st.image(overlay_img, use_container_width=True)