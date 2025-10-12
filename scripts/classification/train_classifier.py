import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

print("--- Starting Stage 1: Training the Classification Model (ResNet50) ---")

# --- Configuration ---
CLASSIFICATION_DATA_DIR = r'D:\CliniScan_Final_Project\classification_dataset'
TRAIN_DIR = os.path.join(CLASSIFICATION_DATA_DIR, 'train')
VAL_DIR = os.path.join(CLASSIFICATION_DATA_DIR, 'val')
RESULTS_DIR = r'D:\CliniScan_Final_Project\training_results\classifier_model_resnet'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 # We can start with 10 epochs and see how it performs

# --- 1. Data Preparation ---
print("\n[1/4] Preparing Data Generators...")
# Create data generators with augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for the validation set (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # Normal vs Abnormal
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 2. Model Building (ResNet50) ---
print("[2/4] Building ResNet50 model...")
# Load the ResNet50 model, pre-trained on ImageNet, without its top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the layers of the base model so we don't retrain them
for layer in base_model.layers:
    layer.trainable = False

# Add our custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Final output layer: 1 neuron with a sigmoid activation for binary (Normal/Abnormal) classification
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Model Compilation & Training ---
print("[3/4] Compiling and Training the model...")
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 4. Saving Results ---
print("[4/4] Saving model and performance graphs...")
os.makedirs(RESULTS_DIR, exist_ok=True)
model.save(os.path.join(RESULTS_DIR, 'best_classifier.h5'))

# Plot and save accuracy graph
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_plot.png'))

# Plot and save loss graph
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_plot.png'))

print(f"\n✅✅✅ Classification Model (ResNet50) Training Complete! ✅✅✅")
print(f"Model and graphs saved in: {RESULTS_DIR}")