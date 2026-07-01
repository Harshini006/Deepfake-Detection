import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

try:
    test_engine = tf.keras.models.load_model('best_deepfake_detector_v2.keras')
    print("Model successfully loaded from checkpoint file!")
except:
    print("Checkpoint 'best_deepfake_detector_v2.keras' needs to be in execution path.")

def predict_any_image_size_calibrated(image_path):
    img_raw = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.15, minNeighbors=6, minSize=(40, 40))
    
    if len(faces) == 0:
        faces = [(0, 0, img_raw.shape[1], img_raw.shape[0])]
        
    output_img = img_rgb.copy()
    
    for i, (x, y, w, h) in enumerate(faces):
        pad_h, pad_w = int(h * 0.25), int(w * 0.25)
        y1, y2 = max(0, y - pad_h), min(img_rgb.shape[0], y + h + pad_h)
        x1, x2 = max(0, x - pad_w), min(img_rgb.shape[1], x + w + pad_w)
        
        face_crop = img_rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (224, 224))
        
        face_array = np.array(face_resized, dtype=np.float32)
        face_tensor = np.expand_dims(face_array, axis=0)
        face_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(face_tensor)
        
        score = test_engine.predict(face_preprocessed, verbose=0)[0][0]
        
        if score > 0.582:
            label = f"FAKE: {((score - 0.582) / (1 - 0.582) * 50 + 50):.1f}%"
            color = (255, 0, 0)
        else:
            real_pct = ((0.582 - score) / 0.582) * 50 + 50
            label = f"REAL: {max(50.1, real_pct):.1f}%"
            color = (0, 255, 0)
            
        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 4)
        cv2.putText(output_img, label, (x, max(0, y-15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
    plt.figure(figsize=(10, 7))
    plt.imshow(output_img)
    plt.axis('off')
    plt.show()
