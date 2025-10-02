import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="CliniScan - Multi-Class", layout="wide")

@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

# Path to your BEST multi-class model from Run 3

model_path = r'D:\CliniScan_Project\yolo_results\final_model_multiclass\weights\best.pt'

try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Error loading the multi-class model from {model_path}. Make sure this training run is complete.")
    st.stop()

st.title("🩺 CliniScan: Multi-Class Abnormality Detection")
st.write("This app uses the multi-class model, which was trained to detect 14 different abnormalities.")
st.write("---")

with st.sidebar:
    st.header("Controls")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    st.info("This model attempts to identify specific types of abnormalities.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with st.spinner('Running inference...'):
        results = model.predict(image, conf=confidence_threshold)
    
    result_image_rgb = Image.fromarray(results[0].plot()[..., ::-1])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Prediction Result")
        st.image(result_image_rgb, use_column_width=True)

    if len(results[0].boxes) == 0:
        st.warning("No abnormalities were detected with the current confidence threshold.")
    else:
        st.success(f"Detected {len(results[0].boxes)} potential abnormalities.")