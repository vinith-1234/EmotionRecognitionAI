import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add root directory to python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Streamlit Page Config
st.set_page_config(
    page_title="AI Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        background: linear-gradient(135deg, #4F46E5, #A855F7);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .confidence-box {
        background: #DCFCE7;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #166534;
        font-weight: 700;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    import tensorflow as tf
    model_path = os.path.join(ROOT_DIR, "saved_models", "emotion_model.keras")
    if not os.path.exists(model_path):
        alt = os.path.join(ROOT_DIR, "saved_models", "emotion_model.h5")
        if os.path.exists(alt):
            model_path = alt
        else:
            return None
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_bgr):
    """Convert BGR image to 48x48 grayscale normalized tensor."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    tensor = np.expand_dims(normalized, axis=-1)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def extract_facial_dynamics(img_bgr):
    """Compute dynamic facial structural features for person-specific predictions."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 10 or w < 10:
        return np.ones(7) / 7.0

    upper = gray[0:int(h * 0.55), :]
    lower = gray[int(h * 0.55):h, :]

    edges_lower = cv2.Canny(lower, 50, 150)
    edge_density = np.mean(edges_lower) / 255.0
    lower_var = np.std(lower)
    upper_var = np.std(upper)
    edges_upper = cv2.Canny(upper, 60, 160)
    upper_edge = np.mean(edges_upper) / 255.0

    scores = np.zeros(7)
    scores[0] = upper_edge * 1.4                                          # Angry
    scores[1] = lower_var / 100.0                                         # Disgust
    scores[2] = edge_density * 0.8 + upper_var / 90.0                    # Fear
    scores[3] = edge_density * 1.8 + lower_var / 80.0                    # Happy
    scores[4] = 1.0 / (1.0 + abs(lower_var - upper_var) / 20.0 + edge_density * 2.0)  # Neutral
    scores[5] = upper_var / 70.0 * 1.1                                    # Sad
    scores[6] = (lower_var / 60.0) * 1.5 if lower.mean() < gray.mean() * 0.9 else 0.2  # Surprise

    exp_s = np.exp(scores - np.max(scores))
    return exp_s / np.sum(exp_s)

def detect_and_predict(img_bgr, model):
    """Run face detection + CNN + dynamics fusion on an image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = []
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            fc = cv2.CascadeClassifier(cascade_path)
            faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    except Exception:
        faces = []

    results = []
    rois = [(0, 0, img_bgr.shape[1], img_bgr.shape[0])] if len(faces) == 0 else faces

    for (x, y, w, h) in rois:
        roi = img_bgr[y:y+h, x:x+w]
        tensor = preprocess_image(roi)
        cnn_preds = model.predict(tensor, verbose=0)[0]
        dyn_preds = extract_facial_dynamics(roi)
        fused = (cnn_preds * 0.65) + (dyn_preds * 0.35)
        fused = fused / np.sum(fused)

        idx = int(np.argmax(fused))
        results.append({
            "box": (x, y, w, h),
            "emotion": EMOTION_LABELS[idx],
            "confidence": float(fused[idx]) * 100,
            "probabilities": {EMOTION_LABELS[i]: float(fused[i]) * 100 for i in range(7)}
        })
        
        # Draw on image
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{EMOTION_LABELS[idx]} ({fused[idx]*100:.1f}%)",
                    (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return results, img_bgr

# ─── App UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎭 AI Facial Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Powered Real-Time Facial Expression Classifier</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 About This App")
    st.info("""
**Model:** Deep CNN (TensorFlow/Keras)  
**Dataset:** FER-2013 (48×48 Grayscale)  
**Emotions:**  
😠 Angry | 🤢 Disgust | 😨 Fear  
😊 Happy | 😐 Neutral | 😢 Sad  
😲 Surprise
    """)
    st.markdown("---")
    st.success("🌱 **SDG 3: Good Health & Well-Being**\n\nAI-driven emotion recognition for mental health monitoring and patient wellbeing tracking.")

# Load model
with st.spinner("Loading Deep Learning model..."):
    model = load_model_cached()

if model is None:
    st.error("⚠️ Trained model not found. Please ensure `saved_models/emotion_model.keras` is committed to the repository.")
    st.stop()
else:
    st.sidebar.success("✅ Model Loaded Successfully")

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🖼️ Upload Image", "📷 Live Camera Snapshot"])

# TAB 1: Upload Image
with tab1:
    st.subheader("Upload a Facial Image for Emotion Analysis")
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)

        if st.button("🚀 Analyze Emotion", key="btn_analyze"):
            with st.spinner("Running CNN inference..."):
                results, annotated = detect_and_predict(img.copy(), model)

            with col2:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Output", use_container_width=True)

            top = results[0]
            st.markdown(f'<div class="emotion-box">🎭 {top["emotion"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-box">Confidence: {top["confidence"]:.1f}%</div>', unsafe_allow_html=True)

            st.subheader("📊 Probability Distribution")
            df = pd.DataFrame({
                "Emotion": list(top["probabilities"].keys()),
                "Probability (%)": [round(v, 2) for v in top["probabilities"].values()]
            }).sort_values("Probability (%)", ascending=False)
            st.bar_chart(df.set_index("Emotion"))
            st.dataframe(df, use_container_width=True)

# TAB 2: Live Camera
with tab2:
    st.subheader("📷 Capture a Live Photo for Emotion Detection")
    st.info("Click **Take Photo** below. The app will analyze the captured facial expression.")

    camera_img = st.camera_input("Take a photo")

    if camera_img:
        bytes_data = camera_img.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analyzing captured face..."):
            results, annotated = detect_and_predict(img.copy(), model)

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
        with col_b:
            top = results[0]
            st.markdown(f'<div class="emotion-box">🎭 {top["emotion"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-box">Confidence: {top["confidence"]:.1f}%</div>', unsafe_allow_html=True)

            st.subheader("Probability Breakdown")
            df = pd.DataFrame({
                "Emotion": list(top["probabilities"].keys()),
                "Probability (%)": [round(v, 2) for v in top["probabilities"].values()]
            }).sort_values("Probability (%)", ascending=False)
            st.bar_chart(df.set_index("Emotion"))
