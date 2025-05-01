#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for dental measurements.

This module implements functions to visualize tooth detections and
measurements on dental radiograph images.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def visualize_measurements(image, measurements, show_labels=True):
    """Visualize tooth measurements on the input image.
    
    Args:
        image (numpy.ndarray): Input radiograph image
        measurements (list): List of measurement results
        show_labels (bool): Whether to show measurement labels
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Make a copy of the image to draw on
    if len(image.shape) == 2:  # If grayscale
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    # Define colors for different tooth types
    primary_color = (0, 255, 0)  # Green for primary molars
    premolar_color = (0, 0, 255)  # Blue for premolars
    line_color = (255, 0, 0)  # Red for measurement lines
    
    # Draw tooth contours and measurements
    for result in measurements:
        primary = result['primary_molar']
        premolar = result['premolar']
        width_diff = result['width_difference']
        
        # Draw primary molar
        if primary:
            tooth = primary['tooth']
            cv2.drawContours(vis_img, [tooth.contour], 0, primary_color, 2)
            
            # Draw contact points and width line
            left, right = primary['contact_points']
            cv2.circle(vis_img, left, 3, line_color, -1)
            cv2.circle(vis_img, right, 3, line_color, -1)
            cv2.line(vis_img, left, right, line_color, 1)
            
            if show_labels:
                # Add width label
                label_pos = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2 - 10)
                cv2.putText(vis_img, f"{primary['width']:.2f} mm", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, primary_color, 1)
        
        # Draw premolar
        if premolar:
            tooth = premolar['tooth']
            cv2.drawContours(vis_img, [tooth.contour], 0, premolar_color, 2)
            
            # Draw contact points and width line
            left, right = premolar['contact_points']
            cv2.circle(vis_img, left, 3, line_color, -1)
            cv2.circle(vis_img, right, 3, line_color, -1)
            cv2.line(vis_img, left, right, line_color, 1)
            
            if show_labels:
                # Add width label
                label_pos = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2 + 15)
                cv2.putText(vis_img, f"{premolar['width']:.2f} mm", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, premolar_color, 1)
        
        # Add difference label
        if primary and premolar and show_labels:
            # Midpoint between the two teeth
            primary_center = primary['tooth'].centroid
            premolar_center = premolar['tooth'].centroid
            diff_label_pos = ((primary_center[0] + premolar_center[0]) // 2 + 20,
                             (primary_center[1] + premolar_center[1]) // 2)
            
            cv2.putText(vis_img, f"Diff: {width_diff:.2f} mm", diff_label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return vis_img


def visualize_tooth_detection(image, teeth):
    """Visualize detected teeth with classifications.
    
    Args:
        image (numpy.ndarray): Input radiograph image
        teeth (list): List of Tooth objects
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Make a copy of the image to draw on
    if len(image.shape) == 2:  # If grayscale
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    # Define colors for different tooth types
    colors = {
        "primary_molar": (0, 255, 0),  # Green
        "premolar": (0, 0, 255),      # Blue
        "other": (255, 0, 255)        # Magenta
    }
    
    # Draw contours with different colors based on tooth type
    for tooth in teeth:
        color = colors.get(tooth.type, (255, 255, 255))  # Default to white
        cv2.drawContours(vis_img, [tooth.contour], 0, color, 2)
        
        # Add label for tooth type
        cv2.putText(vis_img, tooth.type, tooth.centroid,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_img


def create_heatmap(image, measurements):
    """Create a heatmap visualization of tooth width differences.
    
    Args:
        image (numpy.ndarray): Input radiograph image
        measurements (list): List of measurement results
        
    Returns:
        numpy.ndarray: Heatmap image
    """
    # Create an empty mask
    if len(image.shape) == 2:  # If grayscale
        height, width = image.shape
    else:
        height, width, _ = image.shape
    
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Fill the mask with width difference values
    for result in measurements:
        primary = result['primary_molar']
        premolar = result['premolar']
        width_diff = result['width_difference']
        
        if primary and premolar:
            primary_contour = primary['tooth'].contour
            premolar_contour = premolar['tooth'].contour
            
            # Fill the regions with width difference values
            cv2.drawContours(mask, [primary_contour], 0, width_diff, -1)
            cv2.drawContours(mask, [premolar_contour], 0, width_diff, -1)
    
    # Normalize the mask for visualization
    if np.max(mask) > 0:
        mask = mask / np.max(mask)
    
    # Create a custom colormap (blue to red)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('diff_cmap', colors, N=256)
    
    # Convert mask to heatmap image
    heatmap = np.uint8(cmap(mask) * 255)
    
    # Convert to BGR for OpenCV
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGBA2BGR)
    
    # Blend with original image
    if len(image.shape) == 2:  # If grayscale
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    result = cv2.addWeighted(image_color, 0.7, heatmap, 0.3, 0)
    
    return result
