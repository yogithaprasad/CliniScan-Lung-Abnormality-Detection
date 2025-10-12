import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

DATASET_DIR = r"D:/CliniScan_Project/dataset_final"
IMAGES_DIR = os.path.join(DATASET_DIR, "train")
ANNOTATIONS_CSV = os.path.join(DATASET_DIR, "annotations", "train.csv")
YOLO_DIR = os.path.join(DATASET_DIR, "yolo_dataset_single_class")
TRAIN_RATIO = 0.85

print("--- Starting SINGLE-CLASS Data Preparation ---")

IMAGES_TRAIN_DIR = os.path.join(YOLO_DIR, "images", "train")
IMAGES_VAL_DIR = os.path.join(YOLO_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(YOLO_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(YOLO_DIR, "labels", "val")
for folder in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
    os.makedirs(folder, exist_ok=True)

df = pd.read_csv(ANNOTATIONS_CSV)
df_usable = df[df['class_name'] != 'No finding'].dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])
valid_image_ids = list(df_usable['image_id'].unique())
print(f"Found {len(valid_image_ids)} unique images with valid annotations that will be processed.")

train_ids, val_ids = train_test_split(valid_image_ids, train_size=TRAIN_RATIO, random_state=42)

def create_yolo_label(row, W, H):
    try:
        cls_id = 0 # Map ALL abnormalities to a single class: 0
        x_min, y_min, x_max, y_max = float(row['x_min']), float(row['y_min']), float(row['x_max']), float(row['y_max'])
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(W, x_max), min(H, y_max)
        if x_max <= x_min or y_max <= y_min: return None
        x_center, y_center = ((x_min + x_max)/2)/W, ((y_min + y_max)/2)/H
        width, height = (x_max - x_min)/W, (y_max - y_min)/H
        return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    except Exception:
        return None

for img_id in tqdm(valid_image_ids, desc="Processing images"):
    img_filename, img_path = f"{img_id}.png", os.path.join(IMAGES_DIR, f"{img_id}.png")
    if not os.path.exists(img_path): continue
    try:
        with Image.open(img_path) as img: W, H = img.size
    except Exception: continue
    
    rows = df_usable[df_usable['image_id'] == img_id]
    lines = {create_yolo_label(row, W, H) for _, row in rows.iterrows()}
    lines.discard(None)

    if lines:
        is_training = img_id in train_ids
        dest_img_dir = IMAGES_TRAIN_DIR if is_training else IMAGES_VAL_DIR
        dest_label_dir = LABELS_TRAIN_DIR if is_training else LABELS_VAL_DIR
        shutil.copy(img_path, os.path.join(dest_img_dir, img_filename))
        with open(os.path.join(dest_label_dir, f"{img_id}.txt"), 'w') as f:
            f.write("\n".join(lines))

yaml_content = f"""
path: {os.path.abspath(YOLO_DIR)}
train: images/train
val: images/val
nc: 1
names: ['Abnormality']
"""
with open(os.path.join(YOLO_DIR, "data.yaml"), 'w') as f: f.write(yaml_content)

print("\n✅ Single-class dataset creation complete!")