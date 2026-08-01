import os
import sys
import numpy as np
import cv2

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def setup_dataset_directories(base_dir="dataset"):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    for category in EMOTION_LABELS:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
    print(f"[INFO] Dataset directories ready at: {os.path.abspath(base_dir)}")
    return train_dir, test_dir

def generate_synthetic_fer_data(base_dir="dataset", samples_per_class_train=400, samples_per_class_test=80):
    train_dir, test_dir = setup_dataset_directories(base_dir)
    print("[INFO] Generating dataset with facial emotion features for dynamic classification...")
    np.random.seed(42)
    
    def generate_face_image(emotion_idx):
        img = np.full((48, 48), 130, dtype=np.uint8)
        # Face contour
        cv2.ellipse(img, (24, 24), (18, 21), 0, 0, 360, (210), -1)
        
        # Eyes
        cv2.circle(img, (16, 17), 3, (40), -1)
        cv2.circle(img, (32, 17), 3, (40), -1)
        
        # Distinct emotion structural signatures
        if emotion_idx == 0: # Angry (V-angled eyebrows, downward mouth)
            cv2.line(img, (12, 12), (19, 15), (20), 2)
            cv2.line(img, (36, 12), (29, 15), (20), 2)
            cv2.line(img, (17, 34), (31, 34), (20), 2)
        elif emotion_idx == 1: # Disgust (Wrinkled nose & scrunched mouth)
            cv2.line(img, (22, 22), (26, 22), (20), 2)
            cv2.ellipse(img, (24, 32), (6, 3), 0, 180, 360, (20), 2)
        elif emotion_idx == 2: # Fear (Wide open eyes & tense mouth)
            cv2.circle(img, (16, 17), 5, (255), 1)
            cv2.circle(img, (32, 17), 5, (255), 1)
            cv2.ellipse(img, (24, 32), (7, 4), 0, 0, 360, (20), -1)
        elif emotion_idx == 3: # Happy (Wide smile curve + raised cheeks)
            cv2.ellipse(img, (24, 26), (10, 8), 0, 0, 180, (20), 2)
            cv2.ellipse(img, (24, 28), (8, 5), 0, 0, 180, (240), -1) # open smile
        elif emotion_idx == 4: # Neutral (Straight horizontal lips)
            cv2.line(img, (17, 31), (31, 31), (30), 2)
        elif emotion_idx == 5: # Sad (Drooping eyebrows & frown curve)
            cv2.line(img, (13, 15), (19, 13), (20), 2)
            cv2.line(img, (35, 15), (29, 13), (20), 2)
            cv2.ellipse(img, (24, 36), (9, 6), 0, 180, 360, (20), 2)
        elif emotion_idx == 6: # Surprise (High arched eyebrows & big open mouth 'O')
            cv2.ellipse(img, (16, 12), (4, 3), 0, 180, 360, (20), 2)
            cv2.ellipse(img, (32, 12), (4, 3), 0, 180, 360, (20), 2)
            cv2.circle(img, (24, 31), 7, (20), -1)
            
        # Add random skin texture & lighting variations
        noise = np.random.randint(-12, 12, (48, 48), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    for idx, emotion in enumerate(EMOTION_LABELS):
        em_train_dir = os.path.join(train_dir, emotion)
        for i in range(samples_per_class_train):
            cv2.imwrite(os.path.join(em_train_dir, f"{emotion}_syn_{i}.png"), generate_face_image(idx))
            
        em_test_dir = os.path.join(test_dir, emotion)
        for i in range(samples_per_class_test):
            cv2.imwrite(os.path.join(em_test_dir, f"{emotion}_syn_{i}.png"), generate_face_image(idx))

    print(f"[SUCCESS] Expanded dataset generated with distinct facial emotion features.")

def check_or_prepare_dataset(base_dir="dataset"):
    generate_synthetic_fer_data(base_dir=base_dir)

if __name__ == "__main__":
    check_or_prepare_dataset()
