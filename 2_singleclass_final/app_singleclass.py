import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="CliniScan Lung Abnormality Detection", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    model = YOLO(model_path)
    return model

# --- THIS IS THE CORRECTED PATH ---
# Path now correctly points to your final model's folder
model_path = r'D:/CliniScan_Project/yolo_results/final_model_singleclass/weights/best.pt'

# Load the model
try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Error loading the model. Please check if the path is correct.")
    st.error(f"Path being used: {model_path}")
    st.error(f"Details: {e}")
    st.stop()

# --- UI COMPONENTS ---
st.title("🩺 CliniScan: Lung Abnormality Detection (Single-Class Model)")
st.write("Upload a chest X-ray image. The AI model will detect potential abnormalities.")
st.write("---")

with st.sidebar:
    st.header("Controls")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    st.write("---")
    st.info("This application uses a YOLOv8s model trained to detect a single 'Abnormality' class.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner('Running inference...'):
        results = model.predict(image, conf=confidence_threshold)

    result_image_bgr = results[0].plot()
    result_image_rgb = Image.fromarray(result_image_bgr[..., ::-1])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Prediction Result")
        st.image(result_image_rgb, use_column_width=True)

    if len(results[0].boxes) == 0:
        st.warning("No abnormalities were detected with the current confidence threshold. Try lowering the threshold slider.")
    else:
        st.success(f"Detected {len(results[0].boxes)} potential abnormalities.")