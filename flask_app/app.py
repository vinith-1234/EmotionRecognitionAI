import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add root project directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict_image import predict_single_image, load_emotion_model

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pre-load model globally
print("[INFO] Initializing Flask server & loading Emotion Recognition Model...")
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "emotion_model.keras"))
    model = load_emotion_model(model_path)
    print("[SUCCESS] Deep Learning Model loaded into Flask application context.")
except Exception as e:
    print(f"[WARNING] Could not pre-load model into Flask: {e}")
    model = None

def get_active_model():
    global model
    if model is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "emotion_model.keras"))
        model = load_emotion_model(model_path)
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_emotion_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for upload'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            active_model = get_active_model()
            results, original_img = predict_single_image(filepath, model=active_model)
            
            # Draw bounding box and label on annotated output image
            annotated_img = original_img.copy()
            for res in results:
                (x, y, w, h) = res['box']
                emotion = res['emotion']
                conf = res['confidence']
                
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion} ({conf:.1f}%)"
                cv2.putText(annotated_img, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
            annotated_filename = "annotated_" + filename
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            cv2.imwrite(annotated_filepath, annotated_img)
            
            primary_result = results[0] if results else None
            
            return jsonify({
                'success': True,
                'image_url': f'/uploads/{filename}',
                'annotated_image_url': f'/uploads/{annotated_filename}',
                'top_emotion': primary_result['emotion'] if primary_result else 'Unknown',
                'confidence': round(primary_result['confidence'], 2) if primary_result else 0.0,
                'probabilities': primary_result['probabilities'] if primary_result else {},
                'num_faces_detected': len(results)
            })
            
        except Exception as e:
            return jsonify({'error': f'Prediction execution failed: {str(e)}'}), 500
            
    return jsonify({'error': 'File format not supported. Use JPG, PNG, or WEBP'}), 400

@app.route('/predict_frame', methods=['POST'])
def predict_frame_route():
    """
    Receives a base64 encoded image frame captured directly from the browser's live webcam feed.
    """
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No frame image data provided'}), 400
            
        image_data = data['image_data']
        # Strip header if present (e.g. data:image/jpeg;base64,)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode frame image'}), 400
            
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live_cam_temp.jpg')
        cv2.imwrite(temp_path, img)
        
        active_model = get_active_model()
        results, original_img = predict_single_image(temp_path, model=active_model)
        
        annotated_img = original_img.copy()
        for res in results:
            (x, y, w, h) = res['box']
            emotion = res['emotion']
            conf = res['confidence']
            
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion} ({conf:.1f}%)"
            cv2.putText(annotated_img, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        primary_result = results[0] if results else None
        
        return jsonify({
            'success': True,
            'annotated_b64': f'data:image/jpeg;base64,{annotated_b64}',
            'top_emotion': primary_result['emotion'] if primary_result else 'Unknown',
            'confidence': round(primary_result['confidence'], 2) if primary_result else 0.0,
            'probabilities': primary_result['probabilities'] if primary_result else {},
            'num_faces_detected': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Live camera prediction failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("[INFO] Starting Flask Server at http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
