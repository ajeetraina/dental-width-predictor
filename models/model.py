import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def create_segmentation_model():
    """
    Create a basic segmentation model using OpenCV for initial implementation
    Later can be replaced with proper deep learning model
    """
    class OpenCVSegmentation:
        def __init__(self):
            self.initialized = True
        
        def predict(self, image):
            """
            Basic tooth segmentation using OpenCV
            """
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Edge detection
            edges = cv2.Canny(enhanced, 50, 150)
            
            # Morphological operations to clean up
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create segmentation mask
            mask = np.zeros_like(gray)
            
            # Filter contours by area and aspect ratio (tooth-like shapes)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum tooth area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:  # Tooth-like aspect ratio
                        valid_contours.append(contour)
            
            # Draw valid contours on mask
            cv2.drawContours(mask, valid_contours, -1, 255, -1)
            
            # Convert to 3-class format expected by the system
            # 0: Background, 1: Primary Molar, 2: Premolar
            result = np.zeros((mask.shape[0], mask.shape[1], 3))
            result[:,:,0] = (mask == 0).astype(float)  # Background
            result[:,:,1] = (mask == 255).astype(float) * 0.7  # Primary molars (assumption)
            result[:,:,2] = (mask == 255).astype(float) * 0.3  # Premolars (assumption)
            
            return result
    
    return OpenCVSegmentation()

def create_tooth_classifier():
    """
    Create a basic tooth classifier
    Can start with rule-based approach, later replace with CNN
    """
    class BasicToothClassifier:
        def __init__(self):
            self.classes = ['background', 'primary_molar', 'premolar', 'other']
        
        def predict(self, tooth_regions):
            """
            Basic classification based on size and position
            """
            predictions = []
            
            for region in tooth_regions:
                if len(region) == 0:
                    predictions.append('background')
                    continue
                
                # Calculate region properties
                area = len(region)
                
                # Basic classification logic based on size
                # This is a placeholder - replace with proper ML model
                if area > 1000:
                    predictions.append('primary_molar')
                elif area > 500:
                    predictions.append('premolar')
                else:
                    predictions.append('other')
            
            return predictions
    
    return BasicToothClassifier()

def extract_tooth_measurements(segmentation_mask, calibration_factor=0.1):
    """
    Extract mesiodistal measurements from segmentation mask
    """
    measurements = []
    
    # Convert mask to binary if needed
    if len(segmentation_mask.shape) == 3:
        # Take the tooth channels (1 and 2)
        tooth_mask = segmentation_mask[:,:,1] + segmentation_mask[:,:,2]
        tooth_mask = (tooth_mask > 0.5).astype(np.uint8) * 255
    else:
        tooth_mask = segmentation_mask
    
    # Find contours
    contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum area threshold
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate mesiodistal width (width of bounding box)
            mesiodistal_width_pixels = w
            mesiodistal_width_mm = mesiodistal_width_pixels * calibration_factor
            
            # Get the center position
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Determine quadrant (basic classification)
            img_height, img_width = tooth_mask.shape[:2]
            if center_x < img_width // 2 and center_y < img_height // 2:
                quadrant = "upper_left"
            elif center_x >= img_width // 2 and center_y < img_height // 2:
                quadrant = "upper_right"
            elif center_x < img_width // 2 and center_y >= img_height // 2:
                quadrant = "lower_left"
            else:
                quadrant = "lower_right"
            
            measurement = {
                "position": quadrant,
                "mesiodistal_width_mm": mesiodistal_width_mm,
                "mesiodistal_width_pixels": mesiodistal_width_pixels,
                "center_x": center_x,
                "center_y": center_y,
                "area": cv2.contourArea(contour),
                "bounding_box": (x, y, w, h)
            }
            
            measurements.append(measurement)
    
    return measurements

def create_cnn_classifier_model(input_shape=(128, 128, 1), num_classes=4):
    """
    Create a simple CNN model for tooth classification
    To be used when enough training data is available
    """
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_unet_segmentation_model(input_shape=(512, 512, 1)):
    """
    Create U-Net model for tooth segmentation
    To be used when enough training data is available
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    # Output layer - 3 classes (background, primary_molar, premolar)
    outputs = layers.Conv2D(3, 1, activation='softmax')(conv5)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Utility functions for data preprocessing
def preprocess_dental_xray(image_path, target_size=(512, 512)):
    """
    Preprocess dental X-ray for model input
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply((image * 255).astype(np.uint8))
    image = image.astype(np.float32) / 255.0
    
    return image

def detect_calibration_markers(image):
    """
    Detect calibration markers in dental X-rays to convert pixels to mm
    Returns calibration factor (mm per pixel)
    """
    # Placeholder implementation
    # In practice, this would look for known markers or rulers in the X-ray
    # For now, return a default calibration factor
    
    # Common calibration: 1mm = ~10 pixels (varies by X-ray equipment)
    default_calibration = 0.1  # mm per pixel
    
    return default_calibration
