import pandas as pd
import os

# --- CONFIGURATION ---
ANNOTATIONS_CSV = r"D:\CliniScan_Project\dataset_final\annotations\train.csv"
# ---

print("--- Starting Deep Dataset Analysis ---")

if not os.path.exists(ANNOTATIONS_CSV):
    print(f"❌ ERROR: train.csv not found at {ANNOTATIONS_CSV}")
else:
    try:
        # Load the CSV file
        df = pd.read_csv(ANNOTATIONS_CSV)
        print(f"Successfully loaded {ANNOTATIONS_CSV}")

        # --- Filter to only the USABLE data for our detection task ---
        # 1. Remove all rows that are "No finding"
        # 2. Remove all rows that are missing bounding box coordinates
        df_usable = df[df['class_name'] != 'No finding'].dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])
        
        total_annotations = len(df_usable)
        unique_images_with_annotations = df_usable['image_id'].nunique()

        print("\n--- Overall Data Summary ---")
        print(f"Total number of usable bounding boxes (annotations): {total_annotations}")
        print(f"Total number of unique images with at least one abnormality: {unique_images_with_annotations}")
        
        # --- Analysis 1: How many unique images per abnormality class? ---
        print("\n--- Breakdown by Abnormality Class ---")
        print("Counting unique images for each specific abnormality...")
        
        class_counts = df_usable.groupby('class_name')['image_id'].nunique().sort_values(ascending=False)
        
        print(class_counts.to_string())

        # --- Analysis 2: How many images have multiple abnormalities? ---
        print("\n--- Analysis of Images with Multiple Abnormalities ---")
        
        # Count how many annotations (rows) each image_id has
        image_annotation_counts = df_usable['image_id'].value_counts()
        
        # Filter to find images that have more than 1 annotation
        multi_label_images = image_annotation_counts[image_annotation_counts > 1]
        
        num_multi_label_images = len(multi_label_images)
        
        print(f"Number of unique images with 2 or more abnormalities: {num_multi_label_images}")

        print("\n✅ Deep analysis complete.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")