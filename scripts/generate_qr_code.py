import qrcode
import os

# --- CONFIGURATION ---

# Your final, public Streamlit URL
APP_URL = "https://cliniscan-lung-abnormality-detection-yogithaprasad.streamlit.app/" 

# The name of the image file to save
OUTPUT_FILENAME = "clinican_app_qr_code.png"

# Where to save the QR code image
SAVE_DIRECTORY = "assets"

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- QR Code Generator ---")
    
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        
    output_path = os.path.join(SAVE_DIRECTORY, OUTPUT_FILENAME)
    
    print(f"Generating QR code for URL: {APP_URL}")

    try:
        qr_img = qrcode.make(APP_URL)
        qr_img.save(output_path)
        print("\n" + "="*40)
        print(f"✅✅✅ SUCCESS! ✅✅✅")
        print(f"Final QR Code image saved to: {output_path}")
        print("="*40)
    except Exception as e:
        print(f"\nERROR: Could not generate QR code. Details: {e}")