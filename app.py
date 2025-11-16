
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import tensorflow as tf
import json
import os
import io
import pandas as pd
import base64

# -------------------------
# Page config & helpers
# -------------------------
st.set_page_config(page_title="EcoSort-AI", page_icon="♻️", layout="centered")

# small helper to allow downloading dataframe
def get_table_download_link(df: pd.DataFrame, filename="history.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download history as CSV</a>'
    return href

# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource
def load_model_and_mapping():
    # Try common file names
    model_path = None
    for candidate in ["waste_classifier.h5", "waste_classifier.keras", "model.h5", "model.keras"]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError("No model found. Please place 'waste_classifier.h5' in the app folder.")

    model = tf.keras.models.load_model(model_path)

    mapping_path = "class_indices.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            class_indices = json.load(f)
        index_to_class = {v: k for k, v in class_indices.items()}
    else:
        # default mapping fallback
        index_to_class = {0: "Organic", 1: "Recyclable"}

    return model, index_to_class, model_path

try:
    model, index_to_class, model_file_used = load_model_and_mapping()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# -------------------------
# UI: Sidebar
# -------------------------
with st.sidebar:
    st.header("ℹ️ About EcoSort-AI")
    st.write(
        """
        EcoSort-AI classifies waste images into *Organic* or *Recyclable* using a MobileNetV2-based model
        """
    )
    st.markdown("---")
    st.subheader("Model Info")
    st.write(f"- File: `{os.path.basename(model_file_used)}`")
    st.write("- Architecture: MobileNetV2 (transfer learning)")
    st.write("- Input size used in app: 128×128 (configurable in code)")
    st.markdown("---")
    st.subheader("Tips")
    st.write(
        """
        - For best results, crop the image to the waste object and avoid heavy blur.
        """
    )
    st.markdown("---")
    st.caption("Developed by: G Sri Vindhya")

# -------------------------
# Main UI
# -------------------------
st.title("♻️ EcoSort-AI: Smart Waste Classifier")
st.write("Upload or capture a waste image — the model will classify it as **Organic** or **Recyclable** and show a Grad-CAM heatmap.")

# Options
col1, col2 = st.columns([3, 1])
with col2:
    enhance = st.checkbox("🪄 Enhance image (sharpness)", value=False)
    enhance_amount = st.slider("Enhance amount", 1.0, 3.0, 1.6, step=0.1)
    show_gradcam = st.checkbox("🔍 Show Grad-CAM (explainability)", value=True)

# Input methods
st.subheader("Input")
upload, capture = st.columns(2)
with upload:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with capture:
    cam_image = st.camera_input("Or capture from webcam")

# Preprocess utilities
TARGET_SIZE = (128, 128)  # keep in sync with how the model was trained

def preprocess_pil(img: Image.Image, target_size=TARGET_SIZE, enhance_flag=False, enhance_amt=1.6):
    img = img.convert("RGB")
    if enhance_flag:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(enhance_amt)
    # optional small denoise and sharpen chain to help poor inputs
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr

# Prediction function
def predict_from_array(arr):
    # model output expected to be a single sigmoid value
    pred = model.predict(arr)
    # handle shapes: (1,1) or (1,)
    prob = float(np.squeeze(pred))
    # BY CONVENTION: model was trained with class_indices mapping; we'll assume index 1 means 'Recyclable'
    predicted_index = 1 if prob >= 0.5 else 0
    return prob, predicted_index

# Grad-CAM implementation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, classifier_layer_names=None):
    """
    img_array: preprocessed input with shape (1, H, W, C)
    Returns: heatmap resized to input image size (H, W)
    """
    # Find last conv layer automatically if not provided
    if last_conv_layer_name is None:
        # search for a layer with 'conv' in name from the end
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("Could not find a conv layer for Grad-CAM.")
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        # fallback to first Conv2D found
        last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                last_conv_layer_name = last_conv_layer.name
                break
        if last_conv_layer is None:
            raise ValueError("No Conv2D layer found in model for Grad-CAM.")

    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # for binary sigmoid, use the single output

    # compute gradients of the target (loss) wrt conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    # compute guided weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # remove batch dim
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to [0,1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], TARGET_SIZE)
    heatmap = tf.squeeze(heatmap).numpy()
    return heatmap

def overlay_heatmap_on_image(pil_img, heatmap, intensity=0.5, colormap="jet"):
    import matplotlib.cm as cm
    # ensure heatmap shape matches image
    heatmap_resized = np.uint8(255 * heatmap)
    cmap = cm.get_cmap(colormap)
    colored_map = cmap(heatmap_resized / 255.0)[:, :, :3]
    colored_map = Image.fromarray(np.uint8(colored_map * 255)).resize(pil_img.size)
    overlay = Image.blend(pil_img.convert("RGBA"), colored_map.convert("RGBA"), intensity)
    return overlay

# -------------------------
# Handle input and run prediction
# -------------------------
input_image = None
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file)
    except Exception as e:
        st.error("Unable to open uploaded image.")
elif cam_image is not None:
    try:
        # camera_input returns BytesIO-like
        input_image = Image.open(cam_image)
    except Exception as e:
        st.error("Unable to open camera image.")

if input_image is not None:
    st.write("### 🔎 Preview")
    st.image(input_image, use_column_width=True)

    # Preprocess
    pil_for_display, arr = preprocess_pil(input_image, target_size=TARGET_SIZE, enhance_flag=enhance, enhance_amt=enhance_amount)

    # Predict
    prob, predicted_index = predict_from_array(arr)
    predicted_label = index_to_class.get(predicted_index, "Organic" if predicted_index == 0 else "Recyclable")
    # compute confidences
    recyclable_prob = prob
    organic_prob = 1.0 - prob

    # Show result header
    st.markdown("🧩 Prediction Result")
    if predicted_index == 1:
        st.success(f"**{predicted_label}** — Recyclable ♻️")
    else:
        st.success(f"**{predicted_label}** — Organic 🌿")

    st.write(f"Raw model output (sigmoid): `{prob:.4f}`")
    # Show dual probabilities
    st.write(" 📊 Probabilities")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Organic", f"{organic_prob*100:.2f}%")
        p = st.progress(0)
        p.progress(int(organic_prob*100))
    with col_b:
        st.metric("Recyclable", f"{recyclable_prob*100:.2f}%")
        p2 = st.progress(0)
        p2.progress(int(recyclable_prob*100))

    # Grad-CAM
    if show_gradcam:
        st.write(" Grad-CAM (where the model is looking)")
        try:
            heatmap = make_gradcam_heatmap(arr, model)
            overlay = overlay_heatmap_on_image(pil_for_display, heatmap, intensity=0.5)
            cols = st.columns([1,1])
            with cols[0]:
                st.image(pil_for_display, caption="Preprocessed Input", use_column_width=True)
            with cols[1]:
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")

    # Save prediction history in session_state
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "label": predicted_label,
        "predicted_index": int(predicted_index),
        "recyclable_prob": float(recyclable_prob),
        "organic_prob": float(organic_prob)
    })

    # Show history table
    st.markdown("---")
    st.write("### 📝 Prediction History (this session)")
    df = pd.DataFrame(st.session_state.history[::-1])  
    st.dataframe(df)

    st.markdown(get_table_download_link(df, filename="prediction_history.csv"), unsafe_allow_html=True)

else:
    st.info("👆 Please upload an image or capture one with the webcam to classify.")

# Footer
st.markdown("---")


