import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_data_generators(dataset_dir="dataset", target_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads FER-2013 training and testing data using Keras ImageDataGenerator.
    Applies data augmentation to training data to mitigate overfitting.
    
    Data Augmentations:
    - Rescaling pixel values (1./255)
    - Rotation range: ±15 degrees
    - Width/Height shift: ±10%
    - Shear range: 0.1
    - Zoom range: 0.1
    - Horizontal flip: True
    """
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Dataset folders 'train' or 'test' not found under {os.path.abspath(dataset_dir)}")
        
    print(f"[INFO] Configuring ImageDataGenerators for {train_dir} and {test_dir}...")
    
    # Augmentation pipeline for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Rescaling only for validation/testing set
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    
    print(f"[SUCCESS] Loaded {train_generator.samples} training images across {train_generator.num_classes} classes.")
    print(f"[SUCCESS] Loaded {test_generator.samples} test images across {test_generator.num_classes} classes.")
    
    return train_generator, test_generator
