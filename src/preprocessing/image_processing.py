#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocessing module for dental radiographs.

This module contains functions for enhancing dental radiograph images
to improve tooth detection and measurement accuracy.
"""

import cv2
import numpy as np
from skimage import exposure, filters, morphology, util, restoration, feature, transform


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
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with stronger contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise the image with edge preservation
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply histogram stretching to improve contrast
    p2, p98 = np.percentile(denoised, (2, 98))
    stretched = exposure.rescale_intensity(denoised, in_range=(p2, p98))
    
    # Apply gamma correction to enhance bright regions (teeth)
    gamma_corrected = exposure.adjust_gamma(stretched, 0.8)
    
    # Sharpen the image to enhance edges
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(gamma_corrected, -1, kernel)
    
    # Apply morphological operations to enhance teeth structures
    disk_kernel = morphology.disk(1)
    enhanced_structures = morphology.white_tophat(sharpened, disk_kernel)
    result = cv2.addWeighted(sharpened, 1.0, enhanced_structures, 0.5, 0)
    
    return result


def normalize_image(image):
    """Normalize image intensity values to the range [0, 255].
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    # Check if image is already normalized
    if image.max() <= 1.0:
        normalized = image * 255
    else:
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
    # Apply multi-scale edge enhancement
    edges1 = feature.canny(image, sigma=1.5)
    edges2 = feature.canny(image, sigma=2.5)
    
    # Combine edges from different scales
    combined_edges = np.logical_or(edges1, edges2).astype(np.uint8) * 255
    
    # Apply morphological closing to connect broken edges
    kernel = morphology.disk(1)
    closed_edges = morphology.closing(combined_edges, kernel)
    
    # Dilate edges slightly to enhance visibility
    dilated = morphology.dilation(closed_edges, morphology.disk(1))
    
    return dilated


def segment_teeth_region(image):
    """Segment the dental arch region from the background.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Binary mask of the dental region
    """
    # Apply adaptive thresholding for better local contrast handling
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size (keep only large regions that are likely to be part of the dental arch)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area threshold
            filtered_contours.append(contour)
    
    # Create dental arch mask
    mask = np.zeros_like(opening)
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    
    # Apply morphological closing to fill gaps between teeth
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return mask


def enhance_teeth_boundaries(image):
    """Enhance the boundaries between teeth to improve separation.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Image with enhanced tooth boundaries
    """
    # Apply Laplacian filter to detect edges
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Normalize to enhance contrast
    laplacian_norm = normalize_image(laplacian)
    
    # Threshold to get binary edges
    _, binary_edges = cv2.threshold(laplacian_norm, 50, 255, cv2.THRESH_BINARY)
    
    # Enhance the original image with the detected edges
    enhanced = cv2.addWeighted(image, 0.7, binary_edges, 0.3, 0)
    
    return enhanced
