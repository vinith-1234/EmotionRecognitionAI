import cv2
import numpy as np

IMG_SIZE = (48, 48)

def preprocess_frame(img, target_size=IMG_SIZE):
    """
    Preprocesses a single image or video frame:
    1. Grayscale conversion
    2. Histogram equalization (enhances facial features & contrast under varying light)
    3. Resizing to target dimensions (48x48)
    4. Normalization [0.0, 1.0]
    5. Reshaping for model input tensor shape (1, 48, 48, 1)
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is invalid or empty.")
        
    # Convert to grayscale if image has multiple channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img
        
    # Enhance facial feature contrast using CLAHE / Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Resize image to target dimension (48x48)
    resized = cv2.resize(equalized, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel intensity values [0, 255] -> [0.0, 1.0]
    normalized = resized.astype("float32") / 255.0
    
    # Expand dimensions for model prediction batch (1, 48, 48, 1)
    tensor = np.expand_dims(normalized, axis=-1)
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor

def verify_grayscale(img):
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return True
    return False
