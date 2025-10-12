# 🩺 CliniScan AI: AI-Powered Lung Abnormality Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cliniscan-lung-abnormality-detection-yogithaprasad.streamlit.app/)

An end-to-end deep learning portfolio project demonstrating a complete pipeline for detecting and classifying abnormalities in chest X-ray images, built with Python, YOLOv8, TensorFlow, and deployed as an interactive web application with Streamlit.

---

## 🎥 Live Demonstration

A live, deployed version of this application is available for interaction:
**[https://cliniscan-lung-abnormality-detection-yogithaprasad.streamlit.app/](https://cliniscan-lung-abnormality-detection-yogithaprasad.streamlit.app/)**

You can also watch a video walkthrough of the application's features and functionality:
**[Watch the Demo Video Here](D:\CliniScan-Portfolio\assets\web_working_demo_video)**

<br>

<p align="center">
  <img src="assets/clinican_app_qr_code.png" alt="QR Code to Live App" width="200"/>
  <br>
  <em>Scan the QR code to open the live application on your mobile device.</em>
</p>

---

## 📋 Project Overview

CliniScan AI is a proof-of-concept platform designed to showcase the power of deep learning in medical imaging. The primary goal is to provide an intuitive tool that assists in the analysis of chest radiographs by identifying and localizing potential pathological findings.

**This tool is for educational and demonstration purposes only and is not a certified medical device intended for real-world diagnostic use.**

### Key Features:
- **Dual-Model Pipeline:** Utilizes a two-stage process, combining a high-level classifier with a specific object detector for a comprehensive analysis.
- **Interactive Web Interface:** A user-friendly application built with Streamlit that allows for easy image upload and interaction.
- **Model Interpretability:** Implements Grad-CAM to generate heatmaps, providing visual insight into the classification model's decision-making process.
- **Flexible Detection:** Allows users to switch between a general single-class detector and a more specific multi-class detector.

## 🛠️ Technology Stack

- **Backend & Web Framework:** Python, Streamlit
- **Object Detection:** PyTorch, Ultralytics YOLOv8
- **Classification:** TensorFlow, Keras (ResNet50)
- **Data Handling & Image Processing:** Pandas, NumPy, OpenCV, Pillow
- **Visualization:** Grad-CAM, Plotly (for future metric dashboards)
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub (with Git LFS for large model handling)

## 📂 Project Structure

This repository is organized into a professional, modular structure:
- **/app.py:** The main Streamlit application script.
- **/src/:** Contains helper modules, such as `utils.py` for visualization functions.
- **/models/:** Stores the final, trained model weights (`.pt` and `.h5` files).
- **/assets/:** Contains static assets like images for the UI and performance charts.
- **/scripts/:** Includes scripts for one-time tasks like training the models.
- **/.streamlit/config.toml:** Configuration file for the Streamlit theme.
- **/requirements.txt:** A list of all necessary Python packages for easy setup.

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yogithaprasad/CliniScan-Lung-Abnormality-Detection.git
    cd CliniScan-Lung-Abnormality-Detection
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```