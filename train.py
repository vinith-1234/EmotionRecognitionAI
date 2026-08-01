import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils.dataset_downloader import check_or_prepare_dataset
from preprocessing.dataset_loader import load_data_generators
from models.emotion_net import build_emotion_model
from utils.metrics import plot_training_history

def train_emotion_recognition_model(epochs=25, batch_size=64, learning_rate=0.0005):
    """
    Main training execution function.
    """
    print("\n" + "="*60)
    print("      FACIAL EMOTION RECOGNITION AI - MODEL TRAINING      ")
    print("="*60 + "\n")
    
    # 1. Dataset Verification
    check_or_prepare_dataset(base_dir="dataset")
    
    # 2. Load Data Generators
    train_gen, test_gen = load_data_generators(dataset_dir="dataset", batch_size=batch_size)
    
    # 3. Build Model Architecture
    model = build_emotion_model(input_shape=(48, 48, 1), num_classes=7, learning_rate=learning_rate)
    model.summary()
    
    # 4. Prepare Directories for Model Checkpoints
    saved_models_dir = "saved_models"
    os.makedirs(saved_models_dir, exist_ok=True)
    
    keras_model_path = os.path.join(saved_models_dir, "emotion_model.keras")
    h5_model_path = os.path.join(saved_models_dir, "emotion_model.h5")
    
    # 5. Define Training Callbacks
    checkpoint = ModelCheckpoint(
        filepath=keras_model_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # 6. Execute Model Training
    print(f"\n[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=callbacks
    )
    
    # 7. Save Final Models & Plot Training Curves
    print("\n[INFO] Saving final model artifacts...")
    model.save(keras_model_path)
    try:
        model.save(h5_model_path)
    except Exception as e:
        print(f"[NOTE] Could not save H5 format: {e}")
        
    print(f"[SUCCESS] Model successfully saved to: {os.path.abspath(keras_model_path)}")
    
    plot_training_history(history, output_dir="outputs")
    print("\n[COMPLETE] Model training pipeline finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FER-2013 Facial Emotion Recognition Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for Adam optimizer")
    
    args = parser.parse_args()
    train_emotion_recognition_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
