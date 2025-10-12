# =============================================================================
# CliniScan AI: Final Professional Application (Definitive Version)
# Author: Yogitha Prasad
# =============================================================================

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="CliniScan AI", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR LARGER FONTS ---
st.markdown("""
<style>
/* Your CSS from the last working version */
</style>
""", unsafe_allow_html=True)

# --- PATHS (Based on your final, correct file scan) ---
MODELS = {
    "Single-Class Detector": {"path": "models/model_singleclass_detector/weights/best.pt", "run_dir": "assets/singleclass_detector"},
    "Multi-Class Detector": {"path": "models/model_multiclass_detector/weights/best.pt", "run_dir": "assets/multiclass_detector"},
    "Classifier": {"path": "models/model_resnet_classification/best_classifier.h5", "run_dir": "assets/classification"}
}

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_models():
    """Loads all models safely."""
    try:
        detector_single = YOLO(MODELS["Single-Class Detector"]["path"])
        detector_multi = YOLO(MODELS["Multi-Class Detector"]["path"])
        classifier = load_model(MODELS["Classifier"]["path"], compile=False)
        return {"Single-Class Detector": detector_single, "Multi-Class Detector": detector_multi}, classifier
    except Exception as e:
        st.error(f"FATAL ERROR LOADING MODELS: {e}. Please check your file structure and paths.")
        return None, None

@st.cache_data
def generate_grad_cam(_classifier_model, img_array_normalized, last_conv_layer_name="conv5_block3_out"):
    """Generates a Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model([_classifier_model.inputs], [_classifier_model.get_layer(last_conv_layer_name).output, _classifier_model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array_normalized)
        loss = predictions[0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlays a heatmap on an image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    return superimposed_img

# --- UI PAGES ---
def page_about():
    st.title("Welcome to CliniScan AI 🩺")
    st.markdown("### AI-Powered Lung Abnormality Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("assets/about_image.png", use_container_width=True)

    with col2:
        st.markdown("#### Revolutionizing Medical Image Analysis")
        st.markdown("""
        CliniScan AI leverages a sophisticated machine learning pipeline to analyze complex chest X-ray images. It empowers radiologists and healthcare providers to make faster, more effective decisions for better patient outcomes.
        """)
        st.success("⬅️ **Select 'Live Prediction' from the sidebar to begin your analysis!**")

    st.markdown("---")
    st.subheader("Project Workflow & Objective")
    st.markdown("""
    This project follows a standard machine learning lifecycle to ensure a robust and well-documented solution:
    1.  **Data Acquisition & Preprocessing:** Utilized the public VinDr-CXR dataset and developed scripts to parse annotations into a YOLO-compatible structure.
    2.  **Model Development:** Trained and compared multiple models, including a ResNet50 classifier for initial screening and YOLOv8 detectors for localizing abnormalities.
    3.  **Visualization & Deployment:** Integrated the best-performing models into this interactive Streamlit web application, featuring Grad-CAM for interpretability.
    """)

def page_live_prediction(detector_models, classifier_model, chosen_detector_name, confidence_threshold):
    st.header("Live X-Ray Analysis")
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.write("---")
        st.subheader("1. Original Uploaded Image")
        st.image(original_image, use_container_width=True)

        with st.spinner('Analyzing...'):
            img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            img_for_classifier = keras_image.img_to_array(original_image.resize((224, 224)))
            img_norm = np.expand_dims(img_for_classifier, axis=0) / 255.0
            class_pred = classifier_model.predict(img_norm, verbose=0)[0][0]
            class_result = "Abnormal" if class_pred > 0.5 else "Normal"
            
            st.write("---")
            st.subheader("2. Classification Analysis")
            col1, col2 = st.columns(2)
            with col1:
                if class_result == "Abnormal": st.warning(f"### Result: {class_result}\n**Confidence:** `{class_pred:.2%}`")
                else: st.success(f"### Result: {class_result}\n**Confidence:** `{1-class_pred:.2%}`")
            with col2:
                heatmap = generate_grad_cam(classifier_model, img_norm)
                grad_cam_img = overlay_heatmap(img_cv, heatmap)
                st.image(grad_cam_img, channels="BGR", caption="Grad-CAM (Classifier's Focus)", use_container_width=True)

            st.write("---")
            st.subheader("3. Abnormality Detection")
            detector_to_use = detector_models[chosen_detector_name]
            d1, d2 = st.columns([2, 1])
            with d1:
                detect_res = detector_to_use.predict(original_image, conf=confidence_threshold, verbose=False)[0]
                detect_img = detect_res.plot()
                st.image(detect_img, channels="BGR", caption=f"Detections from '{chosen_detector_name}'", use_container_width=True)
            with d2:
                if len(detect_res.boxes) > 0: st.success(f"#### Found {len(detect_res.boxes)} potential abnormalities.")
                else: st.info("#### No specific regions detected at this confidence level.")

def page_dashboard(detector_models):
    st.header("Model Performance Dashboard")
    
    def display_detector_performance(model_name_key):
        st.subheader(f"Detector Model: `{model_name_key}`")
        run_dir = MODELS[model_name_key]["run_dir"]
        cm_path = os.path.join(run_dir, "confusion_matrix.png")
        results_csv = os.path.join(run_dir, "results.csv")
        
        if os.path.exists(cm_path): st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
        else: st.warning(f"Confusion Matrix not found in `{run_dir}`.")
        if os.path.exists(results_csv): st.dataframe(pd.read_csv(results_csv))
        else: st.warning(f"results.csv not found in `{run_dir}`.")

    display_detector_performance("Single-Class Detector")
    st.write("---")
    display_detector_performance("Multi-Class Detector")
    st.write("---")
    
    st.subheader("Classifier Model")
    run_dir = MODELS["Classifier"]["run_dir"]
    acc_path = os.path.join(run_dir, "accuracy_plot.png")
    loss_path = os.path.join(run_dir, "loss_plot.png")
    c1, c2 = st.columns(2)
    if os.path.exists(acc_path): c1.image(acc_path, caption="Classifier Accuracy", use_container_width=True)
    else: c1.error("Accuracy plot not found.")
    if os.path.exists(loss_path): c2.image(loss_path, caption="Loss", use_container_width=True)
    else: c2.error("Loss plot not found.")

# --- MAIN APP ROUTING & SIDEBAR ---
st.sidebar.title("🩺 CliniScan")
page = st.sidebar.radio("Navigation", ["About", "Live Prediction", "Model Dashboard"], help="Use this menu to navigate between pages.")
st.sidebar.write("---")

detector_models, classifier_model = load_models()
if not detector_models or not classifier_model:
    st.stop()

if page == "About":
    page_about()
else:
    with st.sidebar:
        st.header("🔬 Live Prediction Controls")
        chosen_detector_name = st.selectbox("Choose Detector Model", list(detector_models.keys()))
        confidence_threshold = st.slider("Detection Confidence", 0.05, 1.0, 0.25, 0.01)
    
    if page == "Live Prediction":
        page_live_prediction(detector_models, classifier_model, chosen_detector_name, confidence_threshold)
    elif page == "Model Dashboard":
        page_dashboard(detector_models)