import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocessing.dataset_loader import load_data_generators
from utils.metrics import evaluate_and_plot_metrics
from utils.dataset_downloader import check_or_prepare_dataset

def evaluate_saved_model(model_path="saved_models/emotion_model.keras", dataset_dir="dataset"):
    """
    Evaluates saved FER-2013 emotion recognition model on test set.
    """
    print("\n" + "="*60)
    print("      FACIAL EMOTION RECOGNITION AI - MODEL EVALUATION      ")
    print("="*60 + "\n")
    
    check_or_prepare_dataset(base_dir=dataset_dir)
    
    if not os.path.exists(model_path):
        # Fallback to .h5 if .keras not found
        h5_path = model_path.replace(".keras", ".h5")
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            raise FileNotFoundError(f"No trained model found at {model_path}. Please run train.py first!")
            
    print(f"[INFO] Loading saved model from: {os.path.abspath(model_path)}")
    model = load_model(model_path)
    
    # Load Test Data Generator (shuffle=False to maintain label alignment)
    _, test_gen = load_data_generators(dataset_dir=dataset_dir, batch_size=64)
    
    print("[INFO] Running predictions on test dataset...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Evaluate & Plot Metrics
    metrics_summary = evaluate_and_plot_metrics(y_true, y_pred, output_dir="outputs")
    return metrics_summary

if __name__ == "__main__":
    evaluate_saved_model()
