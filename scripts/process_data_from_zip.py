import zipfile
import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm
import io

# --- CONFIGURE YOUR PROJECT ---
# 1. The exact name of your downloaded zip file.
zip_file_name = "E:/vinbigdata-chest-xray-abnormalities-detection.zip" 

# 2. The name for the main output folder where everything will be saved.
output_folder_name = "D:\CliniScan_Project\dataset_final"
# --------------------------------

# --- Do not change below this line ---
script_dir = os.path.dirname(os.path.abspath(__file__))
zip_file_path = os.path.join(script_dir, zip_file_name)
output_base_path = os.path.join(script_dir, output_folder_name)

# Define the final folder structure
output_dir_train = os.path.join(output_base_path, 'train')
output_dir_test = os.path.join(output_base_path, 'test')
output_dir_annotations = os.path.join(output_base_path, 'annotations')

# Create all the necessary output folders
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)
os.makedirs(output_dir_annotations, exist_ok=True)

if not os.path.exists(zip_file_path):
    print(f"❌ ERROR: Zip file not found at '{zip_file_path}'")
else:
    print(f"Found zip file: '{zip_file_name}'")
    print(f"Starting on-the-fly extraction and conversion...")
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get a list of all members in the zip for the progress bar
        all_files = zip_ref.infolist()
        
        for member in tqdm(all_files, desc="Processing files"):
            # Skip directories
            if member.is_dir():
                continue

            # --- Process DICOM images ---
            if 'train/' in member.filename or 'test/' in member.filename:
                # Extract the file into memory, not to disk
                dicom_bytes = zip_ref.read(member.filename)
                
                try:
                    # Read the DICOM data from memory
                    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
                    pixel_array = ds.pixel_array

                    # Normalize
                    if pixel_array.dtype != np.uint8:
                        pixel_array = pixel_array.astype(np.float32)
                        pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
                        pixel_array = np.uint8(pixel_array)

                    img = Image.fromarray(pixel_array)
                    img_resized = img.resize((512, 512), Image.LANCZOS)

                    # Determine where to save the PNG
                    original_filename = os.path.basename(member.filename)
                    base_name = os.path.splitext(original_filename)[0]
                    
                    if 'train/' in member.filename:
                        output_path = os.path.join(output_dir_train, f"{base_name}.png")
                    else: # It's a test image
                        output_path = os.path.join(output_dir_test, f"{base_name}.png")
                    
                    # Save the small PNG file
                    img_resized.save(output_path, "PNG")

                except Exception:
                    # This might fail on non-DICOM files (like .DS_Store), just ignore and continue
                    continue

            # --- Extract CSV Annotation Files ---
            elif member.filename.endswith('.csv'):
                # Extract the CSVs directly to the annotations folder
                zip_ref.extract(member, path=output_dir_annotations)


    print(f"\n✅ SUCCESS! All images have been converted and saved in '{output_folder_name}'.")
    print("   Annotation CSVs have been extracted to the 'annotations' subfolder.")