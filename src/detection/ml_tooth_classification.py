#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML-based tooth classification module for dental radiographs.

This module implements functions to classify detected teeth using
a trained neural network model.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model loader and Tooth class
from models.model import load_trained_models
from src.detection.tooth_classification import Tooth


def extract_tooth_region(image, contour, target_size=(128, 128)):
    """Extract and preprocess a tooth region for classification.
    
    Args:
        image (numpy.ndarray): Input image
        contour (numpy.ndarray): Contour of the tooth
        target_size (tuple): Target size for the classification model
        
    Returns:
        numpy.ndarray: Preprocessed tooth region
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract the region with some margin
    margin = 10
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(image.shape[1], x + w + margin)
    y_max = min(image.shape[0], y + h + margin)
    
    # Extract the region
    tooth_region = image[y_min:y_max, x_min:x_max]
    
    # Ensure grayscale
    if len(tooth_region.shape) == 3:
        tooth_region = cv2.cvtColor(tooth_region, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    resized = cv2.resize(tooth_region, target_size)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Expand dimensions for model input (batch, height, width, channels)
    model_input = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    
    return model_input


def classify_tooth_with_ml(image, contour, model, class_names=['primary_molar', 'premolar', 'other']):
    """Classify a tooth using the ML model.
    
    Args:
        image (numpy.ndarray): Input image
        contour (numpy.ndarray): Contour of the tooth
        model (tf.keras.Model): Classification model
        class_names (list): Names of the classes
        
    Returns:
        str: Predicted tooth type
    """
    # Extract and preprocess the tooth region
    tooth_input = extract_tooth_region(image, contour)
    
    # Run inference
    prediction = model.predict(tooth_input)[0]
    
    # Get the predicted class
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]
    
    # Map to class name
    tooth_type = class_names[class_id]
    
    return tooth_type, confidence


def determine_position(contour, image):
    """Determine the position of a tooth in the dental arch.
    
    Args:
        contour (numpy.ndarray): Contour of the tooth
        image (numpy.ndarray): Input image
        
    Returns:
        str: Position of the tooth ('upper_left', 'upper_right', 'lower_left', 'lower_right')
    """
    height, width = image.shape[:2]
    midline_x = width // 2
    midline_y = height // 2
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return "unknown"
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Determine position
    if cx < midline_x and cy < midline_y:
        position = "upper_left"
    elif cx >= midline_x and cy < midline_y:
        position = "upper_right"
    elif cx < midline_x and cy >= midline_y:
        position = "lower_left"
    else:
        position = "lower_right"
    
    return position


def refine_ml_classification(teeth, image, additional_data=None):
    """Refine tooth classification using the contours_by_class from ML segmentation.
    
    Args:
        teeth (list): List of Tooth objects
        image (numpy.ndarray): Input image
        additional_data (dict): Additional data from ML detection
        
    Returns:
        None: Updates the teeth list in-place
    """
    # If we have ML segmentation data, use it to help refine classification
    if additional_data and additional_data.get("method") == "ml" and "contours_by_class" in additional_data:
        contours_by_class = additional_data["contours_by_class"]
        
        # Create lookup dictionaries for quick contour matching
        tooth_by_contour = {str(tooth.contour.tobytes()): tooth for tooth in teeth}
        
        # Update tooth types based on ML segmentation classes
        # Class 1 is primary molar, Class 2 is premolar
        for class_id, contours in contours_by_class.items():
            tooth_type = "primary_molar" if class_id == 1 else "premolar"
            
            for contour in contours:
                contour_key = str(contour.tobytes())
                if contour_key in tooth_by_contour:
                    tooth_by_contour[contour_key].type = tooth_type
    
    # Perform additional refinement similar to the traditional approach
    # Separate teeth by quadrant
    quadrants = {
        "upper_left": [],
        "upper_right": [],
        "lower_left": [],
        "lower_right": []
    }
    
    for tooth in teeth:
        quadrants[tooth.position].append(tooth)
    
    # Process each quadrant to ensure consistency
    for position, quadrant_teeth in quadrants.items():
        if not quadrant_teeth:
            continue
            
        # Sort teeth by x-coordinate
        if "left" in position:
            quadrant_teeth.sort(key=lambda t: t.centroid[0], reverse=True)  # From midline outward
        else:
            quadrant_teeth.sort(key=lambda t: t.centroid[0])  # From midline outward
        
        # Check for missing pairs
        primary_molars = [t for t in quadrant_teeth if t.type == "primary_molar"]
        premolars = [t for t in quadrant_teeth if t.type == "premolar"]
        
        # If we have primary molars but no premolars, try to find pairs
        if primary_molars and not premolars:
            for primary in primary_molars:
                # Look for a nearby tooth to classify as premolar
                for tooth in quadrant_teeth:
                    if tooth.type != "primary_molar":
                        # Check if they're roughly aligned
                        x_diff = abs(primary.centroid[0] - tooth.centroid[0])
                        y_diff = abs(primary.centroid[1] - tooth.centroid[1])
                        
                        if x_diff < 50 and y_diff > 30:
                            tooth.type = "premolar"
                            break


def classify_teeth_ml(image, contours, model_dir="models", additional_data=None):
    """Classify detected teeth contours by type and position using ML model.
    
    Args:
        image (numpy.ndarray): Preprocessed radiograph image
        contours (list): List of tooth contours
        model_dir (str): Directory where models are saved
        additional_data (dict): Additional data from ML detection
        
    Returns:
        list: List of Tooth objects with classification information
    """
    # Load models if not already loaded
    _, classification_model = load_trained_models(model_dir)
    
    # If model loading failed, fall back to traditional method
    if classification_model is None:
        from src.detection.tooth_classification import classify_teeth as cv_classify_teeth
        return cv_classify_teeth(image, contours)
    
    classified_teeth = []
    
    for contour in contours:
        # Calculate contour properties
        M = cv2.moments(contour)
        if M["m00"] == 0:  # Skip if contour area is zero
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine position
        position = determine_position(contour, image)
        
        # Use ML model to classify the tooth
        tooth_type, confidence = classify_tooth_with_ml(image, contour, classification_model)
        
        # Fall back to size-based classification if confidence is low
        if confidence < 0.7:
            if area > 2000:
                tooth_type = "primary_molar"
            elif area > 1000:
                tooth_type = "premolar"
            else:
                tooth_type = "other"
        
        # Create Tooth object
        tooth = Tooth(
            contour=contour,
            type=tooth_type,
            position=position,
            centroid=(cx, cy),
            area=area,
            bounding_box=(x, y, w, h)
        )
        
        classified_teeth.append(tooth)
    
    # Further refine classification
    refine_ml_classification(classified_teeth, image, additional_data)
    
    return classified_teeth