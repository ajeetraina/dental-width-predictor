#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocessing module for dental radiographs.

This module contains functions for enhancing dental radiograph images
to improve tooth detection and measurement accuracy.
"""

import cv2
import numpy as np
from skimage import exposure, filters


def preprocess_image(image):
    """Preprocess a dental radiograph image for better feature extraction.
    
    Args:
        image (numpy.ndarray): Input radiograph image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Sharpen the image to enhance edges
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened


def normalize_image(image):
    """Normalize image intensity values to the range [0, 255].
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    # Rescale intensity to 0-255 range
    normalized = exposure.rescale_intensity(image, out_range=(0, 255))
    
    # Convert to 8-bit unsigned integer
    normalized = normalized.astype(np.uint8)
    
    return normalized


def enhance_edges(image):
    """Enhance edges in the image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Edge-enhanced image
    """
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize the magnitude
    magnitude = normalize_image(magnitude)
    
    return magnitude


def segment_teeth_region(image):
    """Segment the dental arch region from the background.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Binary mask of the dental region
    """
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opening)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    return mask
