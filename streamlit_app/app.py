import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict_image import predict_single_image, load_emotion_model
from preprocessing.preprocess import preprocess_frame

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Streamlit Page Config
st.set_page_config(
    page_title="AI Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for Streamlit UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #4F46E5;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .sdg-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cached_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "emotion_model.keras"))
    return load_emotion_model(model_path)

def main():
    st.markdown('<div class="main-title">🎭 AI Human Emotion Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">B.Tech Minor Project • Facial Expression Analysis using Deep Learning</div>', unsafe_allow_html=True)

    # Sidebar Navigation & Info
    st.sidebar.title("📌 System Overview")
    st.sidebar.info("""
    **Project Category:** AI / Computer Vision / Deep Learning
    
    **Dataset:** FER-2013 (48x48 Grayscale)
    
    **Architecture:** Deep Convolutional Neural Network (CNN)
    
    **Emotions Classified:**
    - 😠 Angry
    - 🤢 Disgust
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="sdg-badge">🌱 SDG 3: Good Health & Well-being</div>
    <br><small>Supports AI-driven mental health monitoring and patient sentiment tracking.</small>
    """, unsafe_allow_html=True)

    # Load Model
    try:
        model = get_cached_model()
        st.sidebar.success("✅ Neural Network Model Loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Model Load Error: {e}")
        model = None

    # Main Tabs
    tab1, tab2 = st.tabs(["🖼️ Image Upload Predictor", "📹 Camera Feed Predictor"])

    # -------------------------------------------------------------
    # TAB 1: IMAGE UPLOAD PREDICTOR
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Upload Image for Facial Emotion Classification")
        uploaded_file = st.file_uploader("Choose a facial image file...", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            # Convert uploaded file to OpenCV BGR format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Input Image", use_container_width=True)

            if st.button("🚀 Analyze Emotion", key="predict_btn") and model is not None:
                with st.spinner("Processing deep CNN layers..."):
                    # Save temp image for processing function
                    temp_path = "temp_streamlit_upload.jpg"
                    cv2.imwrite(temp_path, img)

                    results, _ = predict_single_image(temp_path, model=model)

                    # Annotate image
                    annotated_img = img.copy()
                    for res in results:
                        (x, y, w, h) = res['box']
                        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(annotated_img, f"{res['emotion']} ({res['confidence']:.1f}%)",
                                    (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    with col2:
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detection Output", use_container_width=True)
                        
                        top_res = results[0]
                        st.success(f"**Predicted Emotion:** {top_res['emotion']} ({top_res['confidence']:.2f}% Confidence)")

                        # Display Probability Bar Chart
                        df = pd.DataFrame({
                            'Emotion': list(top_res['probabilities'].keys()),
                            'Probability (%)': list(top_res['probabilities'].values())
                        }).sort_values(by='Probability (%)', ascending=True)

                        st.bar_chart(df.set_index('Emotion'))

    # -------------------------------------------------------------
    # TAB 2: CAMERA FEED PREDICTOR
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Capture Live Snapshot from Camera")
        camera_image = st.camera_input("Take a photo to analyze facial emotion")

        if camera_image is not None and model is not None:
            bytes_data = camera_image.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            temp_cam_path = "temp_camera.jpg"
            cv2.imwrite(temp_cam_path, img)

            results, _ = predict_single_image(temp_cam_path, model=model)

            annotated_img = img.copy()
            for res in results:
                (x, y, w, h) = res['box']
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(annotated_img, f"{res['emotion']} ({res['confidence']:.1f}%)",
                            (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if os.path.exists(temp_cam_path):
                os.remove(temp_cam_path)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Annotated Result", use_container_width=True)
            with col_b:
                top_res = results[0]
                st.metric(label="Predicted Emotion", value=top_res['emotion'], delta=f"{top_res['confidence']:.1f}% Confidence")
                
                df_cam = pd.DataFrame({
                    'Emotion': list(top_res['probabilities'].keys()),
                    'Probability (%)': list(top_res['probabilities'].values())
                })
                st.dataframe(df_cam, use_container_width=True)

if __name__ == "__main__":
    main()
