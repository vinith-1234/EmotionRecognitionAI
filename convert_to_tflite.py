import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from models.emotion_net import build_emotion_model

print("[INFO] Building EmotionRecognitionCNN architecture...")
model = build_emotion_model()

h5_path = os.path.join("saved_models", "emotion_model.h5")
keras_path = os.path.join("saved_models", "emotion_model.keras")

if os.path.exists(h5_path):
    print(f"[INFO] Loading weights from {h5_path}...")
    model.load_weights(h5_path)
elif os.path.exists(keras_path):
    print(f"[INFO] Loading weights from {keras_path}...")
    model.load_weights(keras_path)
else:
    print("[ERROR] No weight file found.")
    sys.exit(1)

print("[INFO] Converting to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize to reduce memory to ~15MB!
tflite_model = converter.convert()

tflite_path = os.path.join("saved_models", "emotion_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"[SUCCESS] TFLite model generated: {tflite_path} ({size_mb:.2f} MB)")
