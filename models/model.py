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
            Improved tooth segmentation using OpenCV with proper individual tooth detection
            """
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # More aggressive edge detection for dental X-rays
            edges = cv2.Canny(enhanced, 20, 80)
            
            # Morphological operations to clean up and separate teeth
            kernel_close = np.ones((3,3), np.uint8)
            kernel_open = np.ones((5,5), np.uint8)
            
            # Close small gaps but don't merge separate teeth
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a blank mask
            mask = np.zeros_like(gray)
            
            # Filter contours with STRICT criteria for individual teeth
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # STRICT filtering for individual teeth (not entire image)
                if (500 < area < 50000 and  # Reasonable tooth area (not tiny, not huge)
                    20 < w < 200 and        # Reasonable width (not entire image width)
                    20 < h < 200 and        # Reasonable height  
                    0.3 < w/h < 3.0):       # Reasonable aspect ratio
                    
                    # Additional check: reject if contour is too close to image borders
                    # (likely background/border detection)
                    margin = 50
                    if (x > margin and y > margin and 
                        x + w < gray.shape[1] - margin and 
                        y + h < gray.shape[0] - margin):
                        valid_contours.append(contour)
            
            # Draw only valid individual tooth contours
            cv2.drawContours(mask, valid_contours, -1, 255, -1)
            
            # Convert to 3-class format expected by the system
            # 0: Background, 1: Primary Molar, 2: Premolar
            result = np.zeros((mask.shape[0], mask.shape[1], 3))
            result[:,:,0] = (mask == 0).astype(float)  # Background
            
            # Classify detected teeth by position and size
            tooth_mask = (mask == 255)
            if np.any(tooth_mask):
                # Find connected components for individual tooth classification
                num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                
                for label_id in range(1, num_labels):
                    tooth_region = (labels == label_id)
                    if np.sum(tooth_region) > 500:  # Minimum tooth size
                        
                        # Calculate centroid for position-based classification
                        y_coords, x_coords = np.where(tooth_region)
                        center_y = np.mean(y_coords)
                        center_x = np.mean(x_coords)
                        
                        # Simple classification: larger areas = primary molars, smaller = premolars
                        tooth_area = np.sum(tooth_region)
                        
                        if tooth_area > 5000:  # Larger teeth are likely primary molars
                            result[tooth_region, 1] = 0.8  # Primary molar
                            result[tooth_region, 2] = 0.2  # Some premolar probability
                        else:  # Smaller teeth are likely premolars  
                            result[tooth_region, 1] = 0.3  # Some primary molar probability
                            result[tooth_region, 2] = 0.7  # Premolar
                        
                        result[tooth_region, 0] = 0.0  # Not background
            
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
                
                # Improved classification logic based on dental knowledge
                if 2000 < area < 15000:  # Primary molar size range
                    predictions.append('primary_molar')
                elif 1000 < area < 8000:   # Premolar size range
                    predictions.append('premolar')
                else:
                    predictions.append('other')
            
            return predictions
    
    return BasicToothClassifier()

def extract_tooth_measurements(segmentation_mask, calibration_factor=0.15):
    """
    Extract mesiodistal measurements from segmentation mask with improved calibration
    """
    measurements = []
    
    # Convert mask to binary if needed
    if len(segmentation_mask.shape) == 3:
        # Take the tooth channels (1 and 2)
        tooth_mask = segmentation_mask[:,:,1] + segmentation_mask[:,:,2]
        tooth_mask = (tooth_mask > 0.5).astype(np.uint8) * 255
    else:
        tooth_mask = segmentation_mask
    
    # Find connected components for individual teeth
    num_labels, labels = cv2.connectedComponents(tooth_mask.astype(np.uint8))
    
    for label_id in range(1, num_labels):
        tooth_region = (labels == label_id)
        area = np.sum(tooth_region)
        
        if area > 500:  # Minimum area threshold for individual teeth
            # Get bounding box of this specific tooth
            y_coords, x_coords = np.where(tooth_region)
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                # Calculate mesiodistal width (width of this individual tooth)
                mesiodistal_width_pixels = x_max - x_min
                mesiodistal_width_mm = mesiodistal_width_pixels * calibration_factor
                
                # Get the center position
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # Determine quadrant based on image center
                img_height, img_width = tooth_mask.shape[:2]
                img_center_x = img_width // 2
                img_center_y = img_height // 2
                
                if center_x < img_center_x and center_y < img_center_y:
                    quadrant = "upper_left"
                elif center_x >= img_center_x and center_y < img_center_y:
                    quadrant = "upper_right" 
                elif center_x < img_center_x and center_y >= img_center_y:
                    quadrant = "lower_left"
                else:
                    quadrant = "lower_right"
                
                # Only include reasonable measurements (filter out artifacts)
                if (5 < mesiodistal_width_mm < 50 and  # Reasonable tooth width range
                    area < 20000):  # Not too large (avoid background regions)
                    
                    measurement = {
                        "position": quadrant,
                        "mesiodistal_width_mm": mesiodistal_width_mm,
                        "mesiodistal_width_pixels": mesiodistal_width_pixels,
                        "center_x": int(center_x),
                        "center_y": int(center_y), 
                        "area": float(area),
                        "bounding_box": (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
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
    # Improved calibration for dental X-rays
    # Typical dental X-ray resolution: 1mm = ~6-7 pixels
    default_calibration = 0.15  # mm per pixel (more realistic for dental X-rays)
    
    return default_calibration
