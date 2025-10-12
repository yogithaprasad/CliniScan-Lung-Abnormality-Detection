import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

print("--- Grad-CAM Visualization Script ---")

# --- CONFIGURATION (Corrected Paths) ---
# Use forward slashes for base paths for maximum compatibility
# --- CONFIGURATION (DEFINITIVE FIX) ---
# Use only forward slashes for all paths. This is the most robust method.
# --- CONFIGURATION (FINAL CORRECTED PATHS) ---

# Path to the trained classifier model
MODEL_PATH = r"D:/CliniScan_Final_Project/training_runs/classifier_model_resnet/best_classifier.h5"

# THE FIX IS HERE: Use the real path to the image that we just found.
IMAGE_PATH = r"D:/CliniScan_Final_Project/chestXray8_512/val/images/00000052_000.png"

# The output path for the final heatmap image
OUTPUT_IMAGE_PATH = r"D:/CliniScan_Final_Project/grad_cam_result.png"

# Standard layer name for ResNet50
LAST_CONV_LAYER_NAME = "conv5_block3_out"
IMG_SIZE = (224, 224)
# This is a standard layer name for ResNet50
LAST_CONV_LAYER_NAME = "conv5_block3_out"
IMG_SIZE = (224, 224)
# This is a standard layer name for ResNet50
LAST_CONV_LAYER_NAME = "conv5_block3_out"
IMG_SIZE = (224, 224)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load the trained model
    print(f"\n[1/4] Loading model from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load the model. Please check the path.")
        print(f"Details: {e}")
        exit()

    # 2. Load and preprocess the image
    print(f"[2/4] Loading and preprocessing image: {os.path.basename(IMAGE_PATH)}")
    try:
        img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Image not found at the path: {IMAGE_PATH}")
        print("Please ensure the file exists and the path is correct.")
        exit()
        
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 3. Build the Grad-CAM model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )

    # 4. Generate the heatmap
    print("[3/4] Generating heatmap...")
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 5. Superimpose the heatmap on the original image
    print("[4/4] Superimposing heatmap and saving result...")
    img_cv = cv2.imread(IMAGE_PATH)
    img_cv = cv2.resize(img_cv, IMG_SIZE)

    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_cv
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    cv2.imwrite(OUTPUT_IMAGE_PATH, superimposed_img)
    
    print(f"\n✅✅✅ Grad-CAM image saved successfully at: {OUTPUT_IMAGE_PATH} ✅✅✅")