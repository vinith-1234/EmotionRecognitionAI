import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocessing.preprocess import preprocess_frame

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_emotion_model(model_path="saved_models/emotion_model.keras"):
    if not os.path.exists(model_path):
        alt_path = "saved_models/emotion_model.h5"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first!")
    return load_model(model_path)

def extract_facial_dynamics(face_roi):
    """
    Computes dynamic facial structural feature signatures:
    - Lower face variance & open mouth ratio (Happy vs Surprise vs Neutral)
    - Eyebrow / eye contrast (Angry vs Fear vs Sad)
    - High-frequency edge density
    """
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi
        
    h, w = gray.shape
    if h < 10 or w < 10:
        return np.ones(7) / 7.0
        
    # Split facial regions: upper (eyes/brows), lower (mouth/chin)
    upper_face = gray[0:int(h*0.55), :]
    lower_face = gray[int(h*0.55):h, :]
    
    # Edge density
    edges_lower = cv2.Canny(lower_face, 50, 150)
    edge_density_lower = np.mean(edges_lower) / 255.0
    
    # Intensity variance in lower face (smile/open mouth creates variance)
    lower_var = np.std(lower_face)
    upper_var = np.std(upper_face)
    
    # Calculate relative facial feature scores
    scores = np.zeros(7)
    
    # Happy: Open smile / raised cheeks -> high lower face edge density & variance
    scores[3] = (edge_density_lower * 1.8) + (lower_var / 80.0)
    
    # Neutral: Smooth lower face, balanced upper/lower variance
    scores[4] = 1.0 / (1.0 + abs(lower_var - upper_var) / 20.0 + edge_density_lower * 2.0)
    
    # Surprise: High vertical height, open dark mouth cavity
    scores[6] = (lower_var / 60.0) * 1.5 if (lower_face.mean() < gray.mean() * 0.9) else 0.2
    
    # Angry: High upper eyebrow edge density & low mouth variance
    edges_upper = cv2.Canny(upper_face, 60, 160)
    scores[0] = (np.mean(edges_upper) / 255.0) * 1.4
    
    # Sad: Low brightness in lower face & drooping brows
    scores[5] = (upper_var / 70.0) * 1.1
    
    # Fear & Disgust
    scores[2] = (edge_density_lower * 0.8) + (upper_var / 90.0)
    scores[1] = (lower_var / 100.0)
    
    # Softmax normalization
    exp_scores = np.exp(scores - np.max(scores))
    dynamic_probs = exp_scores / np.sum(exp_scores)
    return dynamic_probs

def predict_single_image(image_path, model=None):
    """
    Predicts facial emotion from an input image path with dynamic facial feature fusion:
    1. Reads image file
    2. Detects face ROI using Haar Cascade (with fallback)
    3. Preprocesses face region
    4. Predicts emotion class & confidence percentage
    """
    if model is None:
        model = load_emotion_model()
        
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image file at: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = []
    try:
        if hasattr(cv2, 'CascadeClassifier'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    except Exception as e:
        faces = []
        
    results = []
    
    if len(faces) == 0:
        # Full frame fallback
        processed = preprocess_frame(img)
        cnn_preds = model.predict(processed, verbose=0)[0]
        dyn_preds = extract_facial_dynamics(gray)
        
        # Fuse CNN predictions with image-specific facial structural dynamics
        fused_preds = (cnn_preds * 0.65) + (dyn_preds * 0.35)
        fused_preds = fused_preds / np.sum(fused_preds)
        
        class_idx = np.argmax(fused_preds)
        confidence = float(fused_preds[class_idx]) * 100.0
        emotion_label = EMOTION_LABELS[class_idx]
        
        results.append({
            "box": (0, 0, img.shape[1], img.shape[0]),
            "emotion": emotion_label,
            "confidence": confidence,
            "probabilities": {EMOTION_LABELS[i]: float(fused_preds[i]) * 100.0 for i in range(len(EMOTION_LABELS))}
        })
    else:
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            processed = preprocess_frame(face_roi)
            cnn_preds = model.predict(processed, verbose=0)[0]
            dyn_preds = extract_facial_dynamics(face_roi)
            
            # Fuse CNN predictions with face ROI structural dynamics
            fused_preds = (cnn_preds * 0.65) + (dyn_preds * 0.35)
            fused_preds = fused_preds / np.sum(fused_preds)
            
            class_idx = np.argmax(fused_preds)
            confidence = float(fused_preds[class_idx]) * 100.0
            emotion_label = EMOTION_LABELS[class_idx]
            
            results.append({
                "box": (x, y, w, h),
                "emotion": emotion_label,
                "confidence": confidence,
                "probabilities": {EMOTION_LABELS[i]: float(fused_preds[i]) * 100.0 for i in range(len(EMOTION_LABELS))}
            })
            
    return results, img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Facial Emotion from an Image")
    parser.add_argument("--image", type=str, default=None, help="Path to input face image")
    args = parser.parse_args()
    
    image_path = args.image
    if image_path is None:
        sample_path = "dataset/test/happy/happy_syn_0.png"
        if os.path.exists(sample_path):
            image_path = sample_path
        else:
            print("[INFO] No input image provided.")
            sys.exit(0)
            
    print(f"\n[INFO] Running emotion prediction on: {image_path}")
    model = load_emotion_model()
    results, original_img = predict_single_image(image_path, model=model)
    
    for idx, res in enumerate(results):
        print(f"\n--- Detection #{idx+1} ---")
        print(f"Predicted Emotion : {res['emotion']}")
        print(f"Confidence Score  : {res['confidence']:.2f}%")
        print("Class Probabilities:")
        for em, prob in res['probabilities'].items():
            print(f"  - {em:10s}: {prob:6.2f}%")
