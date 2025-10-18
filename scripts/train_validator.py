# train_validator.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# --- Configuration ---
# This path should point to the parent folder containing 'chest_xray' and 'not_chest_xray'
DATASET_DIR = r"D:\CliniScan-Portfolio\validator_dataset"
MODEL_SAVE_PATH = r"models/chest_xray_validator.h5"

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# --- 1. Prepare the Data ---
print("Preparing data generators...")
# Use ImageDataGenerator for loading and augmenting data
datagen = ImageDataGenerator(
    rescale=1./255,          # Rescale pixel values to [0, 1]
    rotation_range=15,       # Augmentation to make the model more robust
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,   # X-rays have a clear orientation
    fill_mode='nearest',
    validation_split=0.2     # Use 20% of the data for validation
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',     # Binary classification (chest vs. not-chest)
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- 2. Build the Model ---
print("Building the model...")
# Load MobileNetV2 pre-trained on ImageNet, without its top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the base model so they don't get re-trained initially
for layer in base_model.layers:
    layer.trainable = False

# Add our custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout to prevent overfitting
# Final output layer: 1 neuron with sigmoid for binary classification
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Compile the Model ---
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.0001), # Use a low learning rate for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. Train the Model ---
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,  # 10 epochs is usually enough for fine-tuning
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Save the Model ---
print("Training complete. Saving the model...")
# Ensure the 'models' directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)

print(f"\n\nValidator model saved successfully to: {MODEL_SAVE_PATH}")
# This is important! Note the output to know what 0 and 1 mean.
print("CRITICAL: Note the Class Indices below for use in the app:")
print(train_generator.class_indices)