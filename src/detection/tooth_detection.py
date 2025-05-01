#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tooth detection module for dental radiographs.

This module implements functions to detect and segment individual teeth
in dental radiograph images.
"""

import cv2
import numpy as np
from skimage import feature, morphology, measure, segmentation, filters, exposure


def detect_teeth(image):
    """Detect and segment individual teeth in a preprocessed radiograph.
    
    Args:
        image (numpy.ndarray): Preprocessed radiograph image
        
    Returns:
        list: List of contours representing detected teeth
    """
    # Enhance contrast for better feature detection
    p2, p98 = np.percentile(image, (2, 98))
    image_enhanced = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    # Multi-scale edge detection for more robust tooth boundary detection
    edges1 = feature.canny(image_enhanced, sigma=1.5)
    edges2 = feature.canny(image_enhanced, sigma=2.5)
    edges = np.logical_or(edges1, edges2)
    
    # Dilate edges to close gaps between tooth boundaries
    dilated = morphology.dilation(edges, morphology.disk(2))
    
    # Fill holes to create tooth masks
    filled = morphology.remove_small_holes(dilated, area_threshold=50)
    
    # Further enhance tooth regions by morphological closing
    closed = morphology.closing(filled, morphology.disk(3))
    
    # Label connected components
    labeled = measure.label(closed)
    
    # Filter components by size and shape to identify potential teeth
    props = measure.regionprops(labeled)
    
    # Use more permissive thresholds to detect more potential teeth
    filtered_labels = []
    for prop in props:
        # Filter by area (size)
        if prop.area < 200 or prop.area > 25000:
            continue
        
        # Filter by shape (teeth tend to be somewhat rectangular/elliptical)
        if prop.eccentricity < 0.1:  # Too circular, unlikely to be a tooth
            continue
            
        # Filter by solidity (ratio of pixels in the region to pixels in the convex hull)
        if prop.solidity < 0.5:  # Too "holey" to be a tooth
            continue
            
        filtered_labels.append(prop.label)
    
    # Extract contours for the filtered regions
    contours = []
    for label in filtered_labels:
        binary = (labeled == label).astype(np.uint8) * 255
        
        # Apply watershed segmentation to separate potentially merged teeth
        distance = filters.distance_transform_edt(binary)
        local_max = feature.peak_local_max(distance, min_distance=20, labels=binary)
        markers = np.zeros_like(binary, dtype=np.int32)
        
        # Only apply watershed if we detect multiple peaks (potential merged teeth)
        if len(local_max) > 1:
            # Place markers at the local maxima
            for i, (x, y) in enumerate(local_max, start=1):
                markers[x, y] = i
            
            # Apply watershed segmentation
            watershed_labels = segmentation.watershed(-distance, markers, mask=binary)
            
            # Extract contours from each watershed region
            for region in range(1, watershed_labels.max() + 1):
                region_mask = (watershed_labels == region).astype(np.uint8) * 255
                region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Only add if contour is valid and has reasonable size
                for cnt in region_contours:
                    if cv2.contourArea(cnt) > 200:
                        contours.append(cnt)
        else:
            # If only one peak, use the original contour
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
    
    # Define approximate region for dental arch with more flexible boundaries
    center_x = width // 2
    arch_top = height // 4        # Extend higher
    arch_bottom = height * 3 // 4  # Extend lower
    arch_width = width * 3 // 4    # Wider arch
    
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
                # Additional filter: check if contour has reasonable aspect ratio for a tooth
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(h) / w if w > 0 else 0
                
                # Teeth typically have aspect ratios between 0.5 and 3.0
                if 0.5 <= aspect_ratio <= 3.5:
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
    max_dims = [0, 0]
    
    # Find the maximum dimensions needed for the mask
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_dims[0] = max(max_dims[0], x + w + 10)  # Add padding
        max_dims[1] = max(max_dims[1], y + h + 10)  # Add padding
    
    # Ensure dimensions are valid
    mask_width = max(max_dims[0], 100)
    mask_height = max(max_dims[1], 100)
    
    # Create masks of appropriate size
    for contour in contours:
        mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
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
            current_area = np.sum(current_mask > 0)
            
            # Prevent division by zero
            if current_area == 0:
                continue
                
            overlap = np.sum(intersection > 0) / current_area
            
            if overlap > threshold:
                # Merge the contours
                merged_mask = cv2.bitwise_or(current_mask, masks[j])
                merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if merged_contours:  # Ensure we found contours in the merged mask
                    # Find the largest contour (in case there are multiple)
                    largest_contour = max(merged_contours, key=cv2.contourArea)
                    current_contour = largest_contour
                    current_mask = merged_mask
                    used.add(j)
        
        merged.append(current_contour)
        used.add(i)
    
    return merged


def refine_tooth_contours(image, contours):
    """Refine tooth contours using active contour models (snakes).
    
    Args:
        image (numpy.ndarray): Input image
        contours (list): List of tooth contours
        
    Returns:
        list: Refined tooth contours
    """
    refined_contours = []
    
    # Create edge map for the entire image to guide active contour
    edges = feature.canny(image, sigma=2)
    edge_map = edges.astype(np.float64)
    
    # Ensure edge map is 0-1 for cv2.findContours
    edge_map = edge_map / np.max(edge_map) if np.max(edge_map) > 0 else edge_map
    
    for contour in contours:
        # Convert contour to numpy array format required for active contour
        snake = np.reshape(contour, (-1, 2))
        
        # Apply active contour model
        try:
            # Use distance from edge as external energy
            distance_map = filters.distance_transform_edt(~edges)
            
            # Create a smoothing effect on the contour
            alpha = 0.1  # Controls elasticity
            beta = 0.1   # Controls rigidity
            gamma = 0.01 # Controls attraction to edge
            
            # Apply snake algorithm for a few iterations to refine the contour
            for _ in range(5):
                # Compute forces for each point in the snake
                for i in range(len(snake)):
                    prev_i = (i - 1) % len(snake)
                    next_i = (i + 1) % len(snake)
                    
                    # Elasticity force (alpha)
                    elastic_force = alpha * (snake[prev_i] + snake[next_i] - 2 * snake[i])
                    
                    # External force (gamma)
                    y, x = snake[i].astype(int)
                    if 0 <= y < edge_map.shape[0] and 0 <= x < edge_map.shape[1]:
                        # Use the gradient of the distance map for external force
                        gx, gy = np.gradient(-distance_map)
                        if 0 <= y < gx.shape[0] and 0 <= x < gx.shape[1]:
                            external_force = gamma * np.array([gx[y, x], gy[y, x]])
                            snake[i] = snake[i] + elastic_force + external_force
            
            # Convert back to contour format
            refined_contour = np.reshape(snake, (-1, 1, 2)).astype(np.int32)
            refined_contours.append(refined_contour)
        except Exception as e:
            # Fallback to original contour if refinement fails
            refined_contours.append(contour)
    
    return refined_contours
