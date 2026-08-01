import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def evaluate_and_plot_metrics(y_true, y_pred, output_dir="outputs"):
    """
    Computes Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and saves plot/report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*50)
    print("           MODEL EVALUATION SUMMARY           ")
    print("="*50)
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  Weighted Precision: {precision * 100:.2f}%")
    print(f"  Weighted Recall   : {recall * 100:.2f}%")
    print(f"  Weighted F1 Score : {f1 * 100:.2f}%")
    print("="*50 + "\n")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, zero_division=0)
    print("Classification Report:\n")
    print(report)
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== FER-2013 EMOTION RECOGNITION EVALUATION REPORT ===\n\n")
        f.write(f"Accuracy : {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall   : {recall * 100:.2f}%\n")
        f.write(f"F1 Score : {f1 * 100:.2f}%\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {os.path.abspath(report_path)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.title("FER-2013 Facial Emotion Recognition - Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Emotion Label", fontsize=12)
    plt.ylabel("True Emotion Label", fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to: {os.path.abspath(cm_path)}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report
    }

def plot_training_history(history, output_dir="outputs"):
    """
    Plots and saves training & validation accuracy and loss curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    ax1.plot(epochs, acc, 'bo-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'ro-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy vs Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Loss Plot
    ax2.plot(epochs, loss, 'bo-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'ro-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss vs Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss (Categorical Crossentropy)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    history_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_path, dpi=300)
    plt.close()
    print(f"[INFO] Training history plots saved to: {os.path.abspath(history_path)}")
