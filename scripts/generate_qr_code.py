import qrcode
import os

# --- CONFIGURATION ---

# IMPORTANT: You must deploy your app to get a public URL.
# For now, we will use the local URL. When you deploy it, you will
# come back here and replace this with your public app link.

# NEW, CORRECT URL (Use the one from YOUR terminal)
APP_URL = "http://192.168.1.9:8501"

# The name of the image file that will be saved
OUTPUT_FILENAME = "clinican_app_qr_code.png"

# Where to save the QR code image
SAVE_DIRECTORY = "assets"

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- QR Code Generator ---")
    
    # Ensure the assets directory exists
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        print(f"Created directory: '{SAVE_DIRECTORY}'")
        
    output_path = os.path.join(SAVE_DIRECTORY, OUTPUT_FILENAME)
    
    print(f"Generating QR code for URL: {APP_URL}")

    try:
        # Create the QR code image object
        qr_img = qrcode.make(APP_URL)
        
        # Save the image file
        qr_img.save(output_path)
        
        print("\n" + "="*40)
        print(f"✅✅✅ SUCCESS! ✅✅✅")
        print(f"QR Code image saved successfully to: {output_path}")
        print("="*40)

    except Exception as e:
        print(f"\n" + "="*40)
        print(f"❌❌❌ ERROR: Could not generate QR code.")
        print(f"Details: {e}")
        print("="*40)