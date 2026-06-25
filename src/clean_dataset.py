import cv2
import os
from pathlib import Path
from PIL import Image

def clean_and_crop_dataset(raw_dir, clean_dir, target_size=(224, 224)):
    """
    Scans a messy folder, verifies images, extracts faces with padding, 
    and saves standardized images to a clean directory.
    """
    # Load OpenCV's pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    supported_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    processed_count = 0
    discarded_count = 0
    
    # Setup clean directories
    Path(clean_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🧹 Scanning: {raw_dir} -> Saving to: {clean_dir}")
    
    # Check if input directory exists
    if not os.path.exists(raw_dir):
        print(f"⚠️ Warning: Directory skipped (does not exist): {raw_dir}")
        return

    for img_path in Path(raw_dir).rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in supported_exts:
            try:
                # 1. Verify file integrity using PIL
                with Image.open(img_path) as img:
                    img.verify()
                
                # 2. Read with OpenCV for face processing
                frame = cv2.imread(str(img_path))
                if frame is None:
                    discarded_count += 1
                    continue
                    
                h, w, _ = frame.shape
                
                # Filter out low-res garbage images
                if h < 100 or w < 100:
                    discarded_count += 1
                    continue
                
                # 3. Detect Face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                
                # If no face is detected by the cascade (since they are already tight crops), 
                # let's save the original frame resized so we don't accidentally lose data!
                if len(faces) == 0:
                    resized_face = cv2.resize(frame, target_size)
                    clean_filename = f"clean_{img_path.stem}_direct.jpg"
                    cv2.imwrite(os.path.join(clean_dir, clean_filename), resized_face)
                    processed_count += 1
                    continue

                for idx, (x, y, fw, fh) in enumerate(faces):
                    # Add 15% padding around the face bounding box
                    pad_w = int(fw * 0.15)
                    pad_h = int(fh * 0.15)
                    
                    y1 = max(0, y - pad_h)
                    y2 = min(h, y + fh + pad_h)
                    x1 = max(0, x - pad_w)
                    x2 = min(w, x + fw + pad_w)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # 4. Standardize dimensions to target model size (224x224)
                    resized_face = cv2.resize(face_crop, target_size)
                    
                    # Save clean crop
                    clean_filename = f"clean_{img_path.stem}_face{idx}.jpg"
                    cv2.imwrite(os.path.join(clean_dir, clean_filename), resized_face)
                    processed_count += 1
                    
            except Exception:
                discarded_count += 1
                continue

    print(f"✅ Finished folder! Processed: {processed_count} | Discarded/Skipped: {discarded_count}\n")


if __name__ == "__main__":
    # Since clean_dataset.py is in the root folder alongside 'Deepfake dataset'
    INPUT_BASE = "Deepfake dataset/deepfake_database"
    OUTPUT_BASE = "dataset_clean"
    
    # 1. Clean the Training Set
    print("\n--- Processing Training Data ---")
    clean_and_crop_dataset(f"{INPUT_BASE}/train/real", f"{OUTPUT_BASE}/train/real")
    clean_and_crop_dataset(f"{INPUT_BASE}/train/df", f"{OUTPUT_BASE}/train/fake")
    
    # 2. Clean the Validation Set
    print("\n--- Processing Validation Data ---")
    clean_and_crop_dataset(f"{INPUT_BASE}/validation/real", f"{OUTPUT_BASE}/validation/real")
    clean_and_crop_dataset(f"{INPUT_BASE}/validation/df", f"{OUTPUT_BASE}/validation/fake")
    
    # 3. Clean the Test Set
    print("\n--- Processing Test Data ---")
    clean_and_crop_dataset(f"{INPUT_BASE}/test/real", f"{OUTPUT_BASE}/test/real")
    clean_and_crop_dataset(f"{INPUT_BASE}/test/df", f"{OUTPUT_BASE}/test/fake")
