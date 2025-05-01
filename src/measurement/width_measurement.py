#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Width measurement module for dental radiographs.

This module implements functions to measure the mesiodistal width
of teeth at their contact points.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist


def measure_tooth_width(image, teeth, calibration_factor=1.0):
    """Measure the mesiodistal width of teeth at contact points.
    
    Args:
        image (numpy.ndarray): Input image
        teeth (list): List of Tooth objects
        calibration_factor (float): Conversion factor from pixels to mm
        
    Returns:
        list: List of paired measurements (primary molar, premolar)
    """
    # Group teeth by position (quadrant)
    teeth_by_quadrant = {}
    for tooth in teeth:
        if tooth.position not in teeth_by_quadrant:
            teeth_by_quadrant[tooth.position] = []
        teeth_by_quadrant[tooth.position].append(tooth)
    
    # Find pairs of primary molars and premolars in each quadrant
    pairs = []
    for position, quadrant_teeth in teeth_by_quadrant.items():
        primary_molars = [t for t in quadrant_teeth if t.type == "primary_molar"]
        premolars = [t for t in quadrant_teeth if t.type == "premolar"]
        
        # Match primary molars with their corresponding premolars based on position
        for primary in primary_molars:
            best_match = None
            min_distance = float('inf')
            
            for premolar in premolars:
                # Check horizontal alignment
                x_diff = abs(primary.centroid[0] - premolar.centroid[0])
                if x_diff < 50:  # Threshold for horizontal alignment
                    distance = abs(primary.centroid[1] - premolar.centroid[1])
                    if distance < min_distance:
                        min_distance = distance
                        best_match = premolar
            
            if best_match:
                # Measure widths
                primary_width = measure_contact_points(image, primary, calibration_factor)
                premolar_width = measure_contact_points(image, best_match, calibration_factor)
                
                pairs.append((primary_width, premolar_width))
    
    return pairs


def measure_contact_points(image, tooth, calibration_factor=1.0):
    """Measure the mesiodistal width of a tooth at contact points.
    
    Args:
        image (numpy.ndarray): Input image
        tooth (Tooth): Tooth object
        calibration_factor (float): Conversion factor from pixels to mm
        
    Returns:
        dict: Measurement data including width and contact points
    """
    # Extract the contour and mask for the tooth
    contour = tooth.contour
    x, y, w, h = tooth.bounding_box
    
    # Create a mask for the tooth
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Find the centroid of the tooth
    cx, cy = tooth.centroid
    
    # Determine tooth orientation (vertical or horizontal)
    if h > w:  # Tooth is taller than wide (typical)
        # For typical orientation, we look for contact points along horizontal axis
        left_points = []
        right_points = []
        
        # Scan horizontally from the centroid
        for y_offset in range(-h//4, h//4 + 1, 2):  # Scan around the middle vertical third
            scan_y = cy + y_offset
            
            # Find leftmost point at this height
            for x_scan in range(cx, x, -1):
                if mask[scan_y, x_scan] > 0 and mask[scan_y, x_scan-1] == 0:
                    left_points.append((x_scan, scan_y))
                    break
            
            # Find rightmost point at this height
            for x_scan in range(cx, x+w):
                if mask[scan_y, x_scan] > 0 and mask[scan_y, x_scan+1] == 0:
                    right_points.append((x_scan, scan_y))
                    break
        
        # Filter to find the widest points
        if left_points and right_points:
            # Convert to numpy arrays for vector operations
            left_points = np.array(left_points)
            right_points = np.array(right_points)
            
            # Calculate distances between all pairs of left and right points
            distances = cdist(left_points, right_points)
            
            # Find the pair with maximum distance
            max_idx = np.unravel_index(np.argmax(distances), distances.shape)
            left_contact = tuple(left_points[max_idx[0]])
            right_contact = tuple(right_points[max_idx[1]])
            
            # Calculate width in pixels
            width_pixels = np.sqrt((right_contact[0] - left_contact[0])**2 + 
                                   (right_contact[1] - left_contact[1])**2)
            
            # Convert to mm using calibration factor
            width_mm = width_pixels * calibration_factor
            
            return {
                'tooth': tooth,
                'width': width_mm,
                'contact_points': (left_contact, right_contact),
                'width_pixels': width_pixels
            }
    
    # If no valid measurements are found
    return None


def get_contour_extreme_points(contour):
    """Get the extreme points of a contour (leftmost, rightmost, etc.)
    
    Args:
        contour (numpy.ndarray): Contour points
        
    Returns:
        tuple: (leftmost, rightmost, topmost, bottommost) points
    """
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    
    return leftmost, rightmost, topmost, bottommost
