# AI-Based Human Emotion Recognition Using Facial Expressions Using Deep Learning

**B.Tech Minor Project** | Artificial Intelligence • Computer Vision • Deep Learning

---

## 📌 Project Overview
This project presents an end-to-end real-time AI application that detects human emotions from facial expressions using a webcam feed and static image uploads. The system utilizes a deep Convolutional Neural Network (CNN) trained on the **FER-2013** dataset (48x48 pixel grayscale facial images) to classify 7 fundamental facial emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

The implementation features:
1. **Deep Learning Core**: Custom Keras CNN model with batch normalization, max pooling, and dropout regularization.
2. **Real-time OpenCV Webcam Feed**: Multi-face detection using Haar Cascades, real-time ROI extraction, and live bounding box/confidence overlay.
3. **Flask Web Application**: Responsive glassmorphism web interface for drag-and-drop image prediction.
4. **Streamlit Web Application**: Interactive dashboard for file analysis and camera input predictions.
5. **Evaluation Suite**: Calculates Accuracy, Precision, Recall, F1 Score, and saves Confusion Matrix visual plots.

---

## 🛠️ Project Directory Structure

```
EmotionRecognitionAI/
├── dataset/                      # FER-2013 dataset (train & test subfolders)
├── preprocessing/                # Data loader, normalization & augmentation
│   ├── __init__.py
│   ├── dataset_loader.py
│   └── preprocess.py
├── models/                       # CNN architecture definitions
│   ├── __init__.py
│   └── emotion_net.py
├── saved_models/                 # Saved model weights (.keras / .h5)
│   └── emotion_model.keras
├── outputs/                      # Saved confusion matrix plots & metrics reports
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── training_history.png
├── utils/                        # Utilities for dataset download & metrics
│   ├── __init__.py
│   ├── dataset_downloader.py
│   └── metrics.py
├── webcam/                       # Real-time webcam module
│   ├── __init__.py
│   └── webcam_predict.py
├── flask_app/                    # Flask web application
│   ├── app.py
│   ├── templates/index.html
│   ├── static/style.css
│   └── static/js/script.js
├── streamlit_app/                # Streamlit web application
│   └── app.py
├── predict_image.py              # Single image CLI prediction script
├── train.py                      # Model training execution script
├── evaluate.py                   # Model evaluation script
├── requirements.txt              # Required dependencies
├── README.md                     # Documentation
└── run_project.py                # Master launcher script
```

---

## ⚙️ Software & Dependency Requirements

- **Operating System**: Windows 11 / Linux / macOS
- **Python**: Version 3.11
- **Libraries**:
  - `tensorflow>=2.12.0`
  - `opencv-python>=4.7.0`
  - `numpy>=1.23.5`
  - `pandas>=1.5.3`
  - `matplotlib>=3.7.1`
  - `seaborn>=0.12.2`
  - `scikit-learn>=1.2.2`
  - `flask>=2.3.2`
  - `streamlit>=1.22.0`
  - `pillow>=9.5.0`

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Master Project Launcher
Launch the interactive terminal menu:
```bash
python run_project.py
```

---

## 💻 Standalone Module Execution

### 1. Dataset Setup
Generate synthetic benchmark dataset or prepare directory structure:
```bash
python utils/dataset_downloader.py
```

### 2. Train Model
Train the deep CNN model for specified epochs:
```bash
python train.py --epochs 25 --batch_size 64
```

### 3. Evaluate Model
Compute accuracy, precision, recall, F1 score, and generate confusion matrix plot:
```bash
python evaluate.py
```

### 4. Single Image Emotion Prediction
Predict emotion for a specific facial image file:
```bash
python predict_image.py --image path/to/face.jpg
```

### 5. Real-Time Webcam Feed
Launch live webcam emotion recognition feed (Press 'q' to quit):
```bash
python webcam/webcam_predict.py
```

### 6. Launch Flask Web Application
Access web dashboard at `http://127.0.0.1:5000`:
```bash
python flask_app/app.py
```

### 7. Launch Streamlit Application
Access Streamlit dashboard at `http://localhost:8501`:
```bash
streamlit run streamlit_app/app.py
```

---

## 🌿 Contribution to SDG Goal 3: Good Health & Well-Being

This project aligns directly with **UN Sustainable Development Goal 3: Good Health and Well-Being** by:
- Enabling non-invasive real-time tracking of patient emotional states in clinical and telehealth settings.
- Assisting mental health professionals with objective, quantitative affect tracking over time.
- Supporting intelligent online learning environments that adapt to student stress or confusion.
