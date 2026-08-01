import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocessing.preprocess import preprocess_frame

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Color mapping for visual display (BGR format)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 140, 255),  # Orange
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Bright Green
    'Neutral': (255, 255, 0),  # Cyan
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 255)  # Yellow
}

def start_webcam_emotion_recognition(model_path="saved_models/emotion_model.keras"):
    """
    Launches real-time webcam feed for facial emotion recognition.
    """
    if not os.path.exists(model_path):
        alt_path = "saved_models/emotion_model.h5"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"[ERROR] Trained model file not found at {model_path}. Please run train.py first!")
            return

    print(f"[INFO] Loading emotion recognition deep learning model from {model_path}...")
    model = load_model(model_path)

    # Load OpenCV Haar Cascade face detector
    face_cascade = None
    try:
        if hasattr(cv2, 'CascadeClassifier'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        print(f"[NOTE] Haar Cascade initialization: {e}")


    # Initialize video capture (0 for default built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam video stream. Please check camera connection.")
        return

    print("\n" + "="*60)
    print("   REAL-TIME FACIAL EMOTION RECOGNITION SYSTEM IS ACTIVE   ")
    print("   Press 'q' key in the video window to quit.              ")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from webcam stream.")
            break

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Multi-scale face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            # Extract Face Region of Interest (ROI)
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess face ROI (48x48, grayscale, normalized)
                tensor = preprocess_frame(face_roi)
                preds = model.predict(tensor, verbose=0)[0]
                
                max_idx = np.argmax(preds)
                emotion = EMOTION_LABELS[max_idx]
                confidence = preds[max_idx] * 100.0
                
                color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                
                # Draw bounding box around detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Label banner background box
                label_text = f"{emotion}: {confidence:.1f}%"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
                
                # Render emotion text
                cv2.putText(
                    frame,
                    label_text,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Display probability breakdown on screen
                y_offset = 30
                cv2.putText(frame, "Emotion Probabilities:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                for idx, em in enumerate(EMOTION_LABELS):
                    prob = preds[idx] * 100.0
                    y_offset += 22
                    bar_w = int(prob * 1.5)
                    # Draw mini probability bar
                    cv2.rectangle(frame, (180, y_offset - 12), (180 + bar_w, y_offset), EMOTION_COLORS[em], -1)
                    cv2.putText(frame, f"{em:10s}: {prob:5.1f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"[EXC] Exception during ROI processing: {e}")

        # Show output window
        cv2.imshow("AI Facial Emotion Recognition System", frame)

        # Exit condition: Press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam feed terminated cleanly.")

if __name__ == "__main__":
    start_webcam_emotion_recognition()
