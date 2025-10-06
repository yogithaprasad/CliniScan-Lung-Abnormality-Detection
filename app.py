import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
# THE FIX IS HERE: Rename the import to avoid name conflicts
from tensorflow.keras.preprocessing import image as keras_image 
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd

st.set_page_config(page_title="CliniScan", layout="wide", initial_sidebar_state="expanded")

# --- PATHS (Based on your scan output) ---
BASE_PROJECT_DIR = "D:/CliniScan_Project"
MODELS = {
    "Single-Class Detector": {
        "path": os.path.join(BASE_PROJECT_DIR, "yolo_results", "final_model_singleclass", "weights", "best.pt"),
        "run_dir": os.path.join(BASE_PROJECT_DIR, "yolo_results", "final_model_singleclass")
    },
    "Multi-Class Detector": {
        "path": os.path.join(BASE_PROJECT_DIR, "yolo_results", "final_model_multiclass", "weights", "best.pt"),
        "run_dir": os.path.join(BASE_PROJECT_DIR, "yolo_results", "final_model_multiclass")
    },
    "Classifier": {
        "path": os.path.join(BASE_PROJECT_DIR, "yolo_results", "classifier_model_resnet", "best_classifier.h5"),
        "run_dir": os.path.join(BASE_PROJECT_DIR, "yolo_results", "classifier_model_resnet")
    }
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
        st.error(f"FATAL ERROR LOADING MODELS: {e}. Please ensure all model files exist at their specified paths.")
        return None, None

@st.cache_data
def generate_grad_cam(_classifier_model, img_array_normalized):
    """Generates a Grad-CAM heatmap."""
    last_conv_layer_name = "conv5_block3_out"
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
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# --- MAIN APP LAYOUT ---
st.title("🩺 CliniScan: AI-Powered Lung Abnormality Analysis")

with st.sidebar:
    st.header("⚙️ Controls")
    chosen_detector_name = st.selectbox("Choose Detector Model", ["Single-Class Detector", "Multi-Class Detector"])
    st.info("Adjust confidence to control detector sensitivity.")
    confidence_threshold = st.slider("Detection Confidence", 0.05, 1.0, 0.25, 0.01)

detector_models, classifier_model = load_models()
if not detector_models or not classifier_model:
    st.stop()

detector_to_use = detector_models[chosen_detector_name]

tab1, tab2 = st.tabs(["**Analysis Tool**", "**Model Performance**"])

with tab1:
    uploaded_file = st.file_uploader("Upload a chest X-ray...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.image(original_image, caption="1. Original Image", use_container_width=True)

        with st.spinner('Analyzing...'):
            # THE FIX IS HERE: Use the renamed import 'keras_image'
            img_for_classifier = keras_image.img_to_array(original_image.resize((224, 224)))
            img_norm = np.expand_dims(img_for_classifier, axis=0) / 255.0
            class_pred = classifier_model.predict(img_norm, verbose=0)[0][0]
            class_result = "Abnormal" if class_pred > 0.5 else "Normal"
            
            with col2:
                st.subheader("2. Classification")
                if class_result == "Abnormal": st.warning(f"**Result:** {class_result} ({class_pred:.2%})")
                else: st.success(f"**Result:** {class_result} ({1-class_pred:.2%})")
                heatmap = generate_grad_cam(classifier_model, img_norm)
                grad_cam_img = overlay_heatmap(img_cv, heatmap)
                st.image(grad_cam_img, channels="BGR", caption="Grad-CAM Heatmap", use_container_width=True)

            with col3:
                st.subheader("3. Detection")
                detect_res = detector_to_use.predict(original_image, conf=confidence_threshold, verbose=False)[0]
                detect_img = detect_res.plot()
                st.image(detect_img, channels="BGR", caption=f"Detections from '{chosen_detector_name}'", use_container_width=True)
                if len(detect_res.boxes) > 0: st.success(f"Found {len(detect_res.boxes)} potential abnormalities.")
                else: st.info("No regions detected at this confidence level.")

with tab2:
    st.header("Performance Overview")
    
    st.subheader(f"Detector Model: `{chosen_detector_name}`")
    run_dir = MODELS[chosen_detector_name]["run_dir"]
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    results_csv = os.path.join(run_dir, "results.csv")
    
    if os.path.exists(cm_path): st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    else: st.warning(f"Confusion Matrix not found in `{run_dir}`")
    if os.path.exists(results_csv): st.dataframe(pd.read_csv(results_csv))
    else: st.warning(f"results.csv not found in `{run_dir}`")

    st.write("---")
    
    st.subheader(f"Classifier Model")
    classifier_run_dir = MODELS["Classifier"]["run_dir"]
    acc_path = os.path.join(classifier_run_dir, "accuracy_plot.png")
    loss_path = os.path.join(classifier_run_dir, "loss_plot.png")
    
    c1, c2 = st.columns(2)
    if os.path.exists(acc_path): c1.image(acc_path, caption="Accuracy", use_container_width=True)
    else: c1.error(f"Accuracy plot not found in `{classifier_run_dir}`")
    if os.path.exists(loss_path): c2.image(loss_path, caption="Loss", use_container_width=True)
    else: c2.error(f"Loss plot not found in `{classifier_run_dir}`")