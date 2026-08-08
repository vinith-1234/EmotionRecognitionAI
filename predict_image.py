import os
import sys
import gc
import argparse
import cv2
import numpy as np

# ── Limit TensorFlow CPU thread and memory overhead for cloud containers ─────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from preprocessing.preprocess import preprocess_frame

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def _rebuild_emotion_model():
    """Rebuilds EmotionRecognitionCNN architecture when fallback is needed."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                          Dropout, BatchNormalization, Activation)
    from tensorflow.keras.optimizers import Adam

    model = Sequential(name="EmotionRecognitionCNN")

    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Head
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', name="emotion_output"))

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_emotion_model(model_path="saved_models/emotion_model.tflite"):
    """
    Loads emotion model into memory. Prefers TFLite format (~5.7MB file, ~15MB RAM usage).
    """
    base_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "saved_models"
    tflite_path = os.path.join(base_dir, "emotion_model.tflite")
    keras_path = os.path.join(base_dir, "emotion_model.keras")
    h5_path = os.path.join(base_dir, "emotion_model.h5")

    # 1. Prefer TFLite (Ultra-fast & lightweight)
    if os.path.exists(tflite_path):
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            gc.collect()
            print(f"[SUCCESS] Ultra-fast TFLite model loaded into memory: {tflite_path}")
            return interpreter
        except Exception as e:
            print(f"[WARNING] TFLite load failed: {e}")

    # 2. Try direct Keras / H5 load
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    for path in [keras_path, h5_path]:
        if os.path.exists(path):
            try:
                model = load_model(path)
                gc.collect()
                print(f"[SUCCESS] Keras Model loaded from: {path}")
                return model
            except Exception as e:
                print(f"[WARNING] Direct load failed for '{path}': {e}")
                break

    # 3. Fallback to architecture rebuild + weights load
    weights_path = h5_path if os.path.exists(h5_path) else keras_path
    if os.path.exists(weights_path):
        try:
            model = _rebuild_emotion_model()
            model.load_weights(weights_path)
            gc.collect()
            print(f"[SUCCESS] Model rebuilt & weights loaded from: {weights_path}")
            return model
        except Exception as e2:
            raise RuntimeError(f"[ERROR] All model load strategies failed: {e2}") from e2

    raise FileNotFoundError("No model file (.tflite, .h5, or .keras) found in saved_models/")


def run_model_inference(model_or_interpreter, input_tensor):
    """Executes model inference on both TFLite Interpreter and Keras Model objects."""
    import tensorflow as tf
    if isinstance(model_or_interpreter, tf.lite.Interpreter):
        inp_details = model_or_interpreter.get_input_details()
        out_details = model_or_interpreter.get_output_details()
        model_or_interpreter.set_tensor(inp_details[0]['index'], input_tensor.astype(np.float32))
        model_or_interpreter.invoke()
        return model_or_interpreter.get_tensor(out_details[0]['index'])[0]
    else:
        return model_or_interpreter.predict(input_tensor, verbose=0)[0]


def extract_facial_dynamics(face_roi):
    """Computes dynamic facial structural feature signatures."""
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi

    h, w = gray.shape
    if h < 10 or w < 10:
        return np.ones(7) / 7.0

    upper_face = gray[0:int(h * 0.55), :]
    lower_face = gray[int(h * 0.55):h, :]

    edges_lower = cv2.Canny(lower_face, 50, 150)
    edge_density_lower = np.mean(edges_lower) / 255.0

    lower_var = np.std(lower_face)
    upper_var = np.std(upper_face)

    scores = np.zeros(7)
    scores[3] = (edge_density_lower * 1.8) + (lower_var / 80.0)
    scores[4] = 1.0 / (1.0 + abs(lower_var - upper_var) / 20.0 + edge_density_lower * 2.0)
    scores[6] = (lower_var / 60.0) * 1.5 if (lower_face.mean() < gray.mean() * 0.9) else 0.2

    edges_upper = cv2.Canny(upper_face, 60, 160)
    scores[0] = (np.mean(edges_upper) / 255.0) * 1.4
    scores[5] = (upper_var / 70.0) * 1.1
    scores[2] = (edge_density_lower * 0.8) + (upper_var / 90.0)
    scores[1] = (lower_var / 100.0)

    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def predict_single_image(image_path, model=None):
    """Predicts facial emotion from an input image path with facial dynamics fusion."""
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
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(detected) > 0:
                faces = detected
    except Exception:
        faces = []

    results = []

    if len(faces) == 0:
        processed = preprocess_frame(img)
        cnn_preds = run_model_inference(model, processed)
        dyn_preds = extract_facial_dynamics(gray)

        fused_preds = (cnn_preds * 0.65) + (dyn_preds * 0.35)
        fused_preds = fused_preds / np.sum(fused_preds)

        class_idx = int(np.argmax(fused_preds))
        confidence = float(fused_preds[class_idx]) * 100.0

        results.append({
            "box": (0, 0, img.shape[1], img.shape[0]),
            "emotion": EMOTION_LABELS[class_idx],
            "confidence": confidence,
            "probabilities": {EMOTION_LABELS[i]: float(fused_preds[i]) * 100.0 for i in range(len(EMOTION_LABELS))}
        })
    else:
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            processed = preprocess_frame(face_roi)
            cnn_preds = run_model_inference(model, processed)
            dyn_preds = extract_facial_dynamics(face_roi)

            fused_preds = (cnn_preds * 0.65) + (dyn_preds * 0.35)
            fused_preds = fused_preds / np.sum(fused_preds)

            class_idx = int(np.argmax(fused_preds))
            confidence = float(fused_preds[class_idx]) * 100.0

            results.append({
                "box": (int(x), int(y), int(w), int(h)),
                "emotion": EMOTION_LABELS[class_idx],
                "confidence": confidence,
                "probabilities": {EMOTION_LABELS[i]: float(fused_preds[i]) * 100.0 for i in range(len(EMOTION_LABELS))}
            })

    gc.collect()
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
