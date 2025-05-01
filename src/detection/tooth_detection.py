#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tooth detection module for dental radiographs.

This module implements functions to detect and segment individual teeth
in dental radiograph images.
"""

import cv2
import numpy as np
from skimage import feature, morphology, measure


def detect_teeth(image):
    """Detect and segment individual teeth in a preprocessed radiograph.
    
    Args:
        image (numpy.ndarray): Preprocessed radiograph image
        
    Returns:
        list: List of contours representing detected teeth
    """
    # Edge detection
    edges = feature.canny(image, sigma=3)
    
    # Dilate edges to close gaps
    dilated = morphology.dilation(edges, morphology.disk(3))
    
    # Fill holes to create tooth masks
    filled = morphology.remove_small_holes(dilated, area_threshold=100)
    
    # Label connected components
    labeled = measure.label(filled)
    
    # Filter components by size to remove noise
    props = measure.regionprops(labeled)
    filtered_labels = [prop.label for prop in props if prop.area > 500 and prop.area < 10000]
    
    # Extract contours for the filtered regions
    contours = []
    for label in filtered_labels:
        binary = (labeled == label).astype(np.uint8) * 255
        cnt, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnt:  # If contours were found
            contours.append(cnt[0])
    
    return contours


def filter_teeth_by_location(image, contours):
    """Filter teeth contours by their location in the dental arch.
    
    Args:
        image (numpy.ndarray): Input image
        contours (list): List of tooth contours
        
    Returns:
        list: Filtered list of tooth contours
    """
    height, width = image.shape[:2]
    dental_arch_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define approximate region for dental arch (can be refined for specific radiograph types)
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
