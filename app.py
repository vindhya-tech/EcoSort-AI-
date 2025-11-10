import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")

st.title("♻️ EcoSort-AI: Smart Waste Classifier")
st.write("Upload a waste image and let AI classify it as **Organic** or **Recyclable** using Deep Learning.")

@st.cache_resource
def load_model_and_mapping():
   
    model_path = "waste_classifier.h5" if os.path.exists("waste_classifier.h5") else "waste_classifier.keras"
    model = tf.keras.models.load_model(model_path)

    mapping_path = "class_indices.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            class_indices = json.load(f)
        index_to_class = {v: k for k, v in class_indices.items()}
    else:
        st.warning("⚠️ class_indices.json not found! Using default mapping.")
        index_to_class = {0: "Organic", 1: "Recyclable"}
    return model, index_to_class

model, index_to_class = load_model_and_mapping()

uploaded_file = st.file_uploader("📤 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prob = model.predict(img_array)[0][0]
    predicted_index = 1 if prob >= 0.5 else 0
    predicted_label = index_to_class[predicted_index]

    confidence = prob if predicted_index == 1 else 1 - prob

    st.markdown("### 🧩 Prediction Result")
    st.success(f"The uploaded waste is **{predicted_label}** ♻️")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    st.progress(float(min(max(confidence, 0.0), 1.0)))

    # Optional: show raw probability
    st.caption(f"Raw model output: {prob:.4f}")

else:
    st.info("👆 Please upload an image to start classification.")
