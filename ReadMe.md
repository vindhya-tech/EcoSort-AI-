♻️ EcoSort-AI — Waste Classification Using Deep Learning

EcoSort-AI is a simple and efficient waste-classification system that identifies whether a given image belongs to Organic or Recyclable (Inorganic) waste.
The project uses MobileNetV2 with transfer learning and a Streamlit web interface that allows users to upload or capture images for real-time predictions.

🚀 Features

Deep learning classification using MobileNetV2

Streamlit Web App with:

Image upload & webcam input

Confidence scores

Grad-CAM heatmap for explainability

Prediction history + CSV download

Lightweight and fast on CPU

Clean UI suitable for demos and internships

🧠 Model

Trained using TensorFlow/Keras

Input size: 128×128

Binary classification (Organic vs Recyclable)

Includes image augmentation for better accuracy

📁 Files Included

app.py – Streamlit interface

waste_classifier.h5 – Trained model

class_indices.json – Label mapping

requirements.txt – Dependencies

▶️ How to Run
pip install -r requirements.txt
streamlit run app.py

🌱 Purpose

Built as part of the Edunet Foundation – AICTE Internship, this project demonstrates real-world use of AI for environmental sustainability and waste management.