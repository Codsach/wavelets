import streamlit as st
import cv2
import numpy as np
import pickle
import pywt
import os
from utils import extract_dwt_features, extract_wavelet_packet_features

st.set_page_config(page_title="Wavelet Face Classifier", layout="wide")

st.title("🧠 Human Face Classification using 2D Wavelets & Wavelet Packets")

# ==========================
# Check if model exists
# ==========================
model_path = "wavelet_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

if os.path.getsize(model_path) == 0:
    st.error("Model file is empty! Retrain the model.")
    st.stop()

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

uploaded_file = st.file_uploader("📤 Upload a Face Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # ======================
    # STEP 1 - Read Original Image
    # ======================
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Step 1: Original Image")
    st.image(original_img, channels="BGR")

    # ======================
    # STEP 2 - Convert to Grayscale
    # ======================
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (128, 128))

    st.subheader("Step 2: Grayscale Conversion")
    st.image(gray_img, clamp=True)

    # ======================
    # STEP 3 - 2D Wavelet Decomposition
    # ======================
    coeffs = pywt.dwt2(gray_img, 'haar')
    LL, (LH, HL, HH) = coeffs

    st.subheader("Step 3: 2D Wavelet Decomposition (Haar)")

    col1, col2 = st.columns(2)
    col1.image(LL, caption="LL - Approximation", clamp=True)
    col2.image(LH, caption="LH - Horizontal Details", clamp=True)

    col3, col4 = st.columns(2)
    col3.image(HL, caption="HL - Vertical Details", clamp=True)
    col4.image(HH, caption="HH - Diagonal Details", clamp=True)

    # ======================
    # STEP 4 - Wavelet Packet Decomposition
    # ======================
    wp = pywt.WaveletPacket2D(data=gray_img, wavelet='haar', mode='symmetric', maxlevel=2)
    nodes = wp.get_level(2, order='natural')

    st.subheader("Step 4: Wavelet Packet Decomposition (Level 2)")

    cols = st.columns(4)
    for i in range(min(4, len(nodes))):
        cols[i].image(nodes[i].data, caption=f"WP Node {i}", clamp=True)

    # ======================
    # STEP 5 - Feature Extraction
    # ======================
    dwt_feat, dwt_names = extract_dwt_features(gray_img)
    wp_feat, wp_names = extract_wavelet_packet_features(gray_img)

    features = np.concatenate([dwt_feat, wp_feat])
    feature_names = dwt_names + wp_names

    import pandas as pd

    st.subheader("Step 5: Extracted Features with Names")

    feature_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Value": features
    })

    st.dataframe(feature_df)

    st.subheader("Step 5: Extracted Feature Vector")
    st.write("Total Features Extracted:", len(features))
    st.write("First 10 Feature Values:", features[:10])

    # ======================
    # STEP 6 - Prediction
    # ======================
    features = features.reshape(1, -1)

    prediction = model.predict(features)
    probs = model.predict_proba(features)
    confidence = np.max(probs) * 100

    st.subheader("Step 6: Final Prediction")
    st.success(f"Prediction: {prediction[0]}")
    st.write(f"Confidence: {confidence:.2f}%")