#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Width measurement module for dental radiographs.

This module implements functions to measure the mesiodistal width
of teeth at their contact points.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from skimage import morphology, feature, draw, filters


def measure_tooth_width(image, teeth, calibration_factor=0.1):
    """Measure the mesiodistal width of teeth at contact points.
    
    Args:
        image (numpy.ndarray): Input image
        teeth (list): List of Tooth objects
        calibration_factor (float): Conversion factor from pixels to mm
        
    Returns:
        list: List of paired measurements (primary molar, premolar)
    """
    # Group teeth by position (quadrant)
    teeth_by_quadrant = {
        "upper_left": [],
        "upper_right": [],
        "lower_left": [],
        "lower_right": []
    }
    
    for tooth in teeth:
        teeth_by_quadrant[tooth.position].append(tooth)
    
    # Find pairs of primary molars and premolars in each quadrant
    pairs = []
    
    # Process each quadrant separately
    for position, quadrant_teeth in teeth_by_quadrant.items():
        primary_molars = [t for t in quadrant_teeth if t.type == "primary_molar"]
        premolars = [t for t in quadrant_teeth if t.type == "premolar"]
        
        # Match primary molars with their corresponding premolars based on position
        for primary in primary_molars:
            best_match = None
            min_distance = float('inf')
            
            for premolar in premolars:
                # Check horizontal alignment (adjusted threshold for better matching)
                x_diff = abs(primary.centroid[0] - premolar.centroid[0])
                if x_diff < 75:  # Increased threshold for horizontal alignment
                    # Compute a sophisticated alignment score based on both x and y distances
                    # We want teeth that are close horizontally but appropriately separated vertically
                    y_diff = abs(primary.centroid[1] - premolar.centroid[1])
                    
                    # Calculate a distance score that prioritizes horizontal alignment but requires some vertical separation
                    alignment_score = x_diff + max(0, 30 - y_diff) * 2  # Penalize if too close vertically
                    
                    if alignment_score < min_distance:
                        min_distance = alignment_score
                        best_match = premolar
            
            if best_match:
                # Measure widths using enhanced contact point detection
                primary_width = measure_contact_points(image, primary, calibration_factor)
                premolar_width = measure_contact_points(image, best_match, calibration_factor)
                
                # Only add valid measurements
                if primary_width and premolar_width:
                    # Add pair with tooth identifiers
                    pair_data = {
                        "primary_molar": {
                            "position": primary.position,
                            "centroid": primary.centroid,
                            "measurement": primary_width
                        },
                        "premolar": {
                            "position": best_match.position,
                            "centroid": best_match.centroid,
                            "measurement": premolar_width
                        },
                        "width_difference": primary_width["width"] - premolar_width["width"]
                    }
                    
                    pairs.append(pair_data)
    
    return pairs


def measure_contact_points(image, tooth, calibration_factor=0.1):
    """Measure the mesiodistal width of a tooth at contact points using improved algorithm.
    
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
    
    # Apply distance transform to find the medial axis
    distance = filters.distance_transform_edt(mask)
    
    # Find the skeleton/medial axis
    skeleton = morphology.skeletonize(mask > 0)
    
    # Get the centroid and orientation of the tooth
    cx, cy = tooth.centroid
    
    # Use PCA to find the principal axes of the tooth
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None  # No valid mask
        
    mean = np.mean(coords, axis=0)
    coords_centered = coords - mean
    cov = np.cov(coords_centered.T)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Major and minor axes
        major_axis = eigenvectors[:, 0]
        minor_axis = eigenvectors[:, 1]
        
        # Determine if tooth is more vertical or horizontal based on aspect ratio and orientation
        aspect_ratio = float(h) / w if w > 0 else float('inf')
        
        if aspect_ratio > 1.2 or abs(major_axis[1]) > abs(major_axis[0]):  
            # Tooth is taller than wide or major axis is more vertical
            # Width should be measured perpendicular to the major axis (along minor axis)
            measure_axis = minor_axis
        else:
            # Tooth is wider than tall
            # Width should be measured along the minor axis (which is more horizontal)
            measure_axis = minor_axis
        
        # Create perpendicular line through centroid
        perpendicular = np.array([-measure_axis[1], measure_axis[0]])  # Perpendicular to measure_axis
        
        # Scale to ensure line crosses tooth boundaries
        max_dim = max(w, h) * 1.5
        start_point = np.array([cx, cy]) - perpendicular * max_dim
        end_point = np.array([cx, cy]) + perpendicular * max_dim
        
        # Convert to integer coordinates for line drawing
        start_point = tuple(np.round(start_point).astype(int))
        end_point = tuple(np.round(end_point).astype(int))
        
        # Draw the perpendicular line on a separate mask
        line_mask = np.zeros_like(mask)
        cv2.line(line_mask, start_point, end_point, 255, 1)
        
        # Find intersection points with tooth boundary
        intersection = cv2.bitwise_and(line_mask, cv2.Canny(mask, 100, 200))
        intersection_points = np.column_stack(np.where(intersection > 0))
        
        if len(intersection_points) >= 2:
            # If we have at least two intersection points, calculate the distance
            # between the farthest two points (to handle multiple intersections)
            if len(intersection_points) > 2:
                # Find the two points with maximum distance
                max_dist = 0
                p1, p2 = None, None
                
                for i in range(len(intersection_points)):
                    for j in range(i+1, len(intersection_points)):
                        dist = euclidean(intersection_points[i], intersection_points[j])
                        if dist > max_dist:
                            max_dist = dist
                            p1, p2 = intersection_points[i], intersection_points[j]
                
                point1 = tuple(p1[::-1])  # Reverse y,x to x,y for OpenCV
                point2 = tuple(p2[::-1])
            else:
                # Just use the two points we found
                point1 = tuple(intersection_points[0][::-1])  # Reverse y,x to x,y for OpenCV
                point2 = tuple(intersection_points[1][::-1])
            
            # Calculate width in pixels
            width_pixels = euclidean(point1, point2)
            
            # Convert to mm using calibration factor
            width_mm = width_pixels * calibration_factor
            
            return {
                'tooth_type': tooth.type,
                'position': tooth.position,
                'width': width_mm,
                'contact_points': (point1, point2),
                'width_pixels': width_pixels
            }
    except np.linalg.LinAlgError:
        # Fallback to simpler method if PCA fails
        pass
    
    # Fallback method: Find the major diameter using convex hull
    try:
        hull = cv2.convexHull(contour)
        max_dist = 0
        point1, point2 = None, None
        
        # Find the two points on the convex hull with maximum distance
        for i in range(len(hull)):
            for j in range(i+1, len(hull)):
                p1 = tuple(hull[i][0])
                p2 = tuple(hull[j][0])
                dist = euclidean(p1, p2)
                
                if dist > max_dist:
                    max_dist = dist
                    point1, point2 = p1, p2
        
        if point1 and point2:
            width_pixels = max_dist
            width_mm = width_pixels * calibration_factor
            
            return {
                'tooth_type': tooth.type,
                'position': tooth.position,
                'width': width_mm,
                'contact_points': (point1, point2),
                'width_pixels': width_pixels
            }
    except:
        pass
    
    # If all methods fail, try a simple method based on bounding box
    try:
        # For horizontally oriented teeth, use width
        # For vertically oriented teeth, use an approximation
        if w > h:
            width_pixels = w
        else:
            # For vertical teeth, estimate width as the shorter dimension plus a factor
            width_pixels = min(w, h) * 1.2
            
        width_mm = width_pixels * calibration_factor
        
        # Create artificial contact points based on bounding box dimensions
        point1 = (x, y + h//2)
        point2 = (x + w, y + h//2)
        
        return {
            'tooth_type': tooth.type,
            'position': tooth.position,
            'width': width_mm,
            'contact_points': (point1, point2),
            'width_pixels': width_pixels,
            'method': 'approximation'  # Indicate this is an approximation
        }
    except:
        # Last resort
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


def analyze_width_ratios(tooth_pairs):
    """Analyze width ratios to identify patterns and outliers.
    
    Args:
        tooth_pairs (list): List of paired measurements
        
    Returns:
        dict: Analysis results including patterns and statistics
    """
    if not tooth_pairs:
        return {
            "average_ratio": 0,
            "std_deviation": 0,
            "outliers": [],
            "valid_pairs": 0
        }
    
    # Calculate width ratios
    ratios = []
    for pair in tooth_pairs:
        if "primary_molar" in pair and "premolar" in pair:
            primary_width = pair["primary_molar"]["measurement"]["width"]
            premolar_width = pair["premolar"]["measurement"]["width"]
            
            if premolar_width > 0:
                ratio = primary_width / premolar_width
                ratios.append({
                    "ratio": ratio,
                    "pair": pair
                })
    
    if not ratios:
        return {
            "average_ratio": 0,
            "std_deviation": 0,
            "outliers": [],
            "valid_pairs": 0
        }
    
    # Calculate statistics
    ratio_values = [r["ratio"] for r in ratios]
    avg_ratio = np.mean(ratio_values)
    std_dev = np.std(ratio_values)
    
    # Identify outliers (more than 2 standard deviations from mean)
    outliers = []
    for ratio_data in ratios:
        if abs(ratio_data["ratio"] - avg_ratio) > 2 * std_dev:
            outliers.append(ratio_data["pair"])
    
    return {
        "average_ratio": avg_ratio,
        "std_deviation": std_dev,
        "outliers": outliers,
        "valid_pairs": len(ratios)
    }
