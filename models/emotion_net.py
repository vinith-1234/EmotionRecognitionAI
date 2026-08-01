from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

def build_emotion_model(input_shape=(48, 48, 1), num_classes=7, learning_rate=0.0005):
    """
    Builds and compiles a Deep Convolutional Neural Network (CNN) optimized for 48x48 FER-2013 grayscale facial images.
    
    Architecture Layer Explanation:
    -------------------------------
    1. Conv2D Layers: Extract local spatial features (edges, textures, facial keypoints like eyes, lips, nose contour).
    2. BatchNormalization: Normalizes layer activations, speeding up convergence and stabilizing gradient flow.
    3. Activation('relu'): Introduces non-linearity, enabling network to learn complex non-linear facial expressions.
    4. MaxPooling2D: Downsamples spatial dimensions by taking maximum values in 2x2 windows, ensuring translation invariance & parameter reduction.
    5. Dropout: Randomly deactivates neurons during training (25% in conv blocks, 50% in dense layers) to prevent overfitting.
    6. Flatten: Converts 2D spatial feature maps into a 1D vector for dense classification layers.
    7. Dense Layers: High-level feature integration across spatial locations.
    8. Softmax Output: Computes probability distribution across all 7 emotion classes.
    """
    model = Sequential(name="EmotionRecognitionCNN")
    
    # -------------------------------------------------------------
    # CONVOLUTION BLOCK 1 (32 -> 64 Filters)
    # -------------------------------------------------------------
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # -------------------------------------------------------------
    # CONVOLUTION BLOCK 2 (128 Filters)
    # -------------------------------------------------------------
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # -------------------------------------------------------------
    # CONVOLUTION BLOCK 3 (256 Filters)
    # -------------------------------------------------------------
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # -------------------------------------------------------------
    # FULLY CONNECTED CLASSIFIER HEAD
    # -------------------------------------------------------------
    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Output Layer (7 Emotion Classes)
    model.add(Dense(num_classes, activation='softmax', name="emotion_output"))
    
    # Compile Model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_emotion_model()
    model.summary()
