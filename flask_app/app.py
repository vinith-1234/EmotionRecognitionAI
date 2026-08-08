import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from predict_image import predict_single_image, load_emotion_model

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Pre-load model globally on startup ───────────────────────────────────────
print("[INFO] Initializing Flask server & loading Emotion Recognition Model...")
MODEL = None
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "emotion_model.keras")

try:
    MODEL = load_emotion_model(MODEL_PATH)
    print("[SUCCESS] Deep Learning Model loaded into Flask application context.")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_emotion_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Model not available: {e}")
    return MODEL


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_emotion_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Use JPG, PNG or WEBP'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        model = get_model()
        results, original_img = predict_single_image(filepath, model=model)

        # Draw bounding boxes + labels on annotated image
        annotated_img = original_img.copy()
        for res in results:
            x, y, w, h = res['box']
            emotion  = res['emotion']
            conf     = res['confidence']
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (99, 102, 241), 3)
            label = f"{emotion}  {conf:.1f}%"
            cv2.putText(annotated_img, label, (x, max(y - 12, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (99, 102, 241), 2)

        annotated_filename = "annotated_" + filename
        annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_filepath, annotated_img)

        # Also return the original image as base64 for immediate preview
        _, orig_buf = cv2.imencode('.jpg', original_img)
        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf).decode('utf-8')

        _, ann_buf = cv2.imencode('.jpg', annotated_img)
        ann_b64 = "data:image/jpeg;base64," + base64.b64encode(ann_buf).decode('utf-8')

        primary = results[0] if results else None

        return jsonify({
            'success': True,
            'original_b64':  orig_b64,
            'annotated_b64': ann_b64,
            'annotated_image_url': f'/uploads/{annotated_filename}',
            'top_emotion':  primary['emotion']     if primary else 'Unknown',
            'confidence':   round(primary['confidence'], 2) if primary else 0.0,
            'probabilities': primary['probabilities'] if primary else {},
            'num_faces':    len(results)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict_frame', methods=['POST'])
def predict_frame_route():
    """Receives base64 webcam frame and returns emotion prediction."""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No frame data provided'}), 400

        image_data = data['image_data']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode camera frame'}), 400

        # Save temp file for processing
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_temp.jpg')
        cv2.imwrite(temp_path, img)

        model = get_model()
        results, original_img = predict_single_image(temp_path, model=model)

        # Draw on annotated copy
        annotated = original_img.copy()
        for res in results:
            x, y, w, h = res['box']
            emotion  = res['emotion']
            conf     = res['confidence']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (99, 102, 241), 3)
            label = f"{emotion}  {conf:.1f}%"
            cv2.putText(annotated, label, (x, max(y - 12, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (99, 102, 241), 2)

        _, buf = cv2.imencode('.jpg', annotated)
        ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        primary = results[0] if results else None

        return jsonify({
            'success': True,
            'annotated_b64': ann_b64,
            'top_emotion':   primary['emotion']       if primary else 'Unknown',
            'confidence':    round(primary['confidence'], 2) if primary else 0.0,
            'probabilities': primary['probabilities'] if primary else {},
            'num_faces':     len(results)
        })

    except Exception as e:
        return jsonify({'error': f'Camera prediction failed: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("[INFO] Starting Flask Server at http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
