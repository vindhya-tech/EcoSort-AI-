import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")

st.title("♻️ EcoSort-AI: Smart Waste Classifier")
st.write("Upload a waste image and let AI classify it as **Organic** or **Recyclable**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Model is under preprocessing.")
else:
    st.info("👆 Please upload an image to continue.")
