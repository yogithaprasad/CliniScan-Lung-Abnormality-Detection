# In D:\CliniScan-Portfolio\src\utils.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2

@st.cache_data
def generate_grad_cam(_classifier_model, img_array_normalized, last_conv_layer_name="conv5_block3_out"):
    """Generates a Grad-CAM heatmap."""
    # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = Model(
        inputs=[_classifier_model.inputs],
        outputs=[_classifier_model.get_layer(last_conv_layer_name).output, _classifier_model.output]
    )

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # The grad_model has two outputs, get them both
        conv_outputs, predictions = grad_model(img_array_normalized)
        # We are interested in the "Abnormal" class prediction
        loss = predictions[0]

    # Get the gradients of the top predicted class with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) # Add epsilon for stability
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays a heatmap on an image."""
    # Resize the heatmap to match the image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Apply a colormap to the heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    
    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    return superimposed_img