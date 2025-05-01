#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML-based tooth detection module for dental radiographs.

This module implements functions to detect and segment individual teeth
in dental radiograph images using a trained segmentation model.
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from skimage import measure
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model loader
from models.model import load_trained_models


# Global variables to store the loaded models
_segmentation_model = None
_classification_model = None


def load_models(model_dir="models"):
    """Load the trained models from disk.
    
    Args:
        model_dir (str): Directory where models are saved
        
    Returns:
        tuple: Loaded segmentation and classification models
    """
    global _segmentation_model, _classification_model
    
    if _segmentation_model is None or _classification_model is None:
        try:
            _segmentation_model, _classification_model = load_trained_models(model_dir)
            print(f"Models loaded from {model_dir}")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Falling back to traditional methods.")
            from src.detection.tooth_detection import detect_teeth as cv_detect_teeth
            return None, None
    
    return _segmentation_model, _classification_model


def preprocess_for_model(image, target_size=(512, 512)):
    """Preprocess an image for input to the segmentation model.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for the model input
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Expand dimensions for model input (batch, height, width, channels)
    model_input = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    
    return model_input, resized


def detect_teeth_ml(image, model_dir="models"):
    """Detect and segment individual teeth in a preprocessed radiograph using ML.
    
    Args:
        image (numpy.ndarray): Preprocessed radiograph image
        model_dir (str): Directory where models are saved
        
    Returns:
        list: List of contours representing detected teeth
        dict: Additional data including segmentation mask and tooth types
    """
    # Load models if not already loaded
    segmentation_model, _ = load_models(model_dir)
    
    # If model loading failed, fall back to traditional method
    if segmentation_model is None:
        from src.detection.tooth_detection import detect_teeth as cv_detect_teeth
        return cv_detect_teeth(image), {"method": "traditional"}
    
    # Preprocess the image for the model
    model_input, resized = preprocess_for_model(image)
    
    # Run inference with the segmentation model
    prediction = segmentation_model.predict(model_input)[0]
    
    # Get the class with the highest probability for each pixel
    segmentation_mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    
    # Create individual masks for each class
    class_masks = {}
    contours_by_class = {}
    
    # Class 0 is background, 1 is primary molar, 2 is premolar
    for class_id in range(1, 3):  # Skip background (class 0)
        # Create binary mask for this class
        class_mask = (segmentation_mask == class_id).astype(np.uint8) * 255
        class_masks[class_id] = class_mask
        
        # Find contours in the mask
        contours, _ = cv2.findContours(
            class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size to remove noise
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum size threshold
                # Scale the contour back to original image size
                if image.shape[:2] != (512, 512):
                    scaled_contour = []
                    for point in contour:
                        scaled_point = point * np.array([
                            [[image.shape[1] / 512, image.shape[0] / 512]]
                        ])
                        scaled_contour.append(scaled_point.astype(np.int32))
                    contour = np.concatenate(scaled_contour)
                
                filtered_contours.append(contour)
        
        contours_by_class[class_id] = filtered_contours
    
    # Combine all tooth contours
    all_contours = []
    for class_id, contours in contours_by_class.items():
        all_contours.extend(contours)
    
    # Additional information for debugging and visualization
    additional_data = {
        "method": "ml",
        "segmentation_mask": segmentation_mask,
        "class_masks": class_masks,
        "contours_by_class": contours_by_class
    }
    
    return all_contours, additional_data


def filter_teeth_by_location(image, contours):
    """Filter teeth contours by their location in the dental arch.
    
    Args:
        image (numpy.ndarray): Input image
        contours (list): List of tooth contours
        
    Returns:
        list: Filtered list of tooth contours
    """
    # For ML-based detection, we rely more on the model's segmentation
    # but still apply basic location filtering to ensure teeth are in dental arch
    
    height, width = image.shape[:2]
    dental_arch_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define approximate region for dental arch
    center_x = width // 2
    arch_top = height // 3
    arch_bottom = height * 2 // 3
    arch_width = width * 2 // 3
    
    # Create dental arch mask (U-shaped)
    cv2.ellipse(dental_arch_mask, 
               (center_x, arch_bottom),
               (arch_width // 2, arch_bottom - arch_top),
               0, 180, 360, 255, -1)
    
    # Filter contours by checking if their center is within the dental arch mask
    filtered_contours = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            if dental_arch_mask[cy, cx] > 0:
                filtered_contours.append(contour)
    
    return filtered_contours


def merge_overlapping_contours(contours, threshold=0.3):
    """Merge contours that have significant overlap.
    
    Args:
        contours (list): List of contours
        threshold (float): Overlap threshold for merging
        
    Returns:
        list: Merged contours
    """
    # This function is the same as in the traditional approach
    
    # If there are fewer than 2 contours, no merging needed
    if len(contours) < 2:
        return contours
    
    # Create binary masks for each contour
    masks = []
    for contour in contours:
        mask = np.zeros((1000, 1000), dtype=np.uint8)  # Assuming max image size
        cv2.drawContours(mask, [contour], 0, 255, -1)
        masks.append(mask)
    
    # Check for overlaps and merge
    merged = []
    used = set()
    
    for i in range(len(contours)):
        if i in used:
            continue
            
        current_contour = contours[i]
        current_mask = masks[i]
        
        for j in range(i+1, len(contours)):
            if j in used:
                continue
                
            # Calculate overlap
            intersection = cv2.bitwise_and(current_mask, masks[j])
            overlap = np.sum(intersection > 0) / np.sum(current_mask > 0)
            
            if overlap > threshold:
                # Merge the contours
                merged_mask = cv2.bitwise_or(current_mask, masks[j])
                merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                current_contour = merged_contours[0]
                current_mask = merged_mask
                used.add(j)
        
        merged.append(current_contour)
        used.add(i)
    
    return merged