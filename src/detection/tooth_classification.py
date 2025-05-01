#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tooth classification module for dental radiographs.

This module implements functions to classify detected teeth into different
types, with a focus on identifying primary second molars and second premolars.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from skimage import measure, feature


@dataclass
class Tooth:
    """Class representing a tooth with its type and geometric properties."""
    contour: np.ndarray
    type: str  # 'primary_molar', 'premolar', etc.
    position: str  # 'upper_left', 'lower_right', etc.
    centroid: Tuple[int, int]
    area: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h


def classify_teeth(image, contours):
    """Classify detected teeth contours by type and position.
    
    Args:
        image (numpy.ndarray): Preprocessed radiograph image
        contours (list): List of tooth contours
        
    Returns:
        list: List of Tooth objects with classification information
    """
    height, width = image.shape[:2]
    midline_x = width // 2
    midline_y = height // 2
    
    classified_teeth = []
    
    # Calculate dental arch guide lines for better classification
    upper_arch_y = height // 3
    lower_arch_y = height * 2 // 3
    
    for contour in contours:
        # Calculate contour properties
        M = cv2.moments(contour)
        if M["m00"] == 0:  # Skip if contour area is zero
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Compute additional shape features for better classification
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Determine tooth position (quadrant)
        if cx < midline_x and cy < midline_y:
            position = "upper_left"
            relative_y = cy / upper_arch_y  # Relative position to upper arch
        elif cx >= midline_x and cy < midline_y:
            position = "upper_right"
            relative_y = cy / upper_arch_y
        elif cx < midline_x and cy >= midline_y:
            position = "lower_left"
            relative_y = cy / lower_arch_y
        else:
            position = "lower_right"
            relative_y = cy / lower_arch_y
        
        # Extract region of interest for texture analysis
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        roi = cv2.bitwise_and(image, image, mask=mask)
        
        # Compute mean and standard deviation of intensity in the ROI
        roi_pixels = roi[mask > 0]
        intensity_mean = np.mean(roi_pixels) if len(roi_pixels) > 0 else 0
        intensity_std = np.std(roi_pixels) if len(roi_pixels) > 0 else 0
        
        # Calculate texture features
        glcm = feature.graycomatrix(roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        
        # Improved classification using multiple features
        # Primary molars tend to be larger, more rounded, and have different texture than premolars
        
        # Initialize default type
        tooth_type = "other"
        
        # Multi-feature classification
        if "upper" in position:
            # Classification criteria for upper jaw teeth
            if area > 2000 and compactness > 0.3 and contrast < 10:
                if relative_y > 0.7:  # Lower in the upper jaw (closer to occlusal plane)
                    tooth_type = "primary_molar"
                else:
                    tooth_type = "premolar"
            elif area > 1000 and area < 2500 and homogeneity > 0.5:
                tooth_type = "premolar"
            elif area > 1500 and intensity_mean > 100:
                if w > h:  # Wider than tall, likely a molar
                    tooth_type = "primary_molar"
                else:
                    tooth_type = "premolar"
        else:
            # Classification criteria for lower jaw teeth
            if area > 2000 and compactness > 0.3 and contrast < 10:
                if relative_y < 0.5:  # Higher in the lower jaw (closer to occlusal plane)
                    tooth_type = "primary_molar"
                else:
                    tooth_type = "premolar"
            elif area > 1000 and area < 2500 and homogeneity > 0.5:
                tooth_type = "premolar"
            elif area > 1500 and intensity_mean > 100:
                if w > h:  # Wider than tall, likely a molar
                    tooth_type = "primary_molar"
                else:
                    tooth_type = "premolar"
        
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
    
    # Further refine classification using relative positions
    refine_classification(classified_teeth, image)
    
    # Ensure we have both primary molars and premolars by adjusting classification if needed
    ensure_tooth_pairs(classified_teeth, image)
    
    return classified_teeth


def refine_classification(teeth, image):
    """Refine tooth classification using relative positions and features.
    
    Args:
        teeth (list): List of Tooth objects
        image (numpy.ndarray): Input image
        
    Returns:
        None: Updates the teeth list in-place
    """
    height, width = image.shape[:2]
    
    # Separate teeth by quadrant
    quadrants = {
        "upper_left": [],
        "upper_right": [],
        "lower_left": [],
        "lower_right": []
    }
    
    for tooth in teeth:
        quadrants[tooth.position].append(tooth)
    
    # Process each quadrant
    for position, quadrant_teeth in quadrants.items():
        if not quadrant_teeth:
            continue
            
        # Sort teeth by vertical position for more accurate pairing
        if "upper" in position:
            quadrant_teeth.sort(key=lambda t: t.centroid[1])  # Sort by y-coordinate
        else:
            quadrant_teeth.sort(key=lambda t: t.centroid[1], reverse=True)
        
        # Sort teeth by horizontal position within each row
        if "left" in position:
            for i in range(len(quadrant_teeth)):
                row_teeth = [t for t in quadrant_teeth if abs(t.centroid[1] - quadrant_teeth[i].centroid[1]) < 50]
                row_teeth.sort(key=lambda t: t.centroid[0], reverse=True)  # From midline outward
        else:
            for i in range(len(quadrant_teeth)):
                row_teeth = [t for t in quadrant_teeth if abs(t.centroid[1] - quadrant_teeth[i].centroid[1]) < 50]
                row_teeth.sort(key=lambda t: t.centroid[0])  # From midline outward
        
        # Find vertically aligned teeth (potential primary molar-premolar pairs)
        for i, tooth1 in enumerate(quadrant_teeth):
            for j, tooth2 in enumerate(quadrant_teeth):
                if i == j:
                    continue
                    
                # Check if they're roughly vertically aligned
                x_diff = abs(tooth1.centroid[0] - tooth2.centroid[0])
                y_diff = abs(tooth1.centroid[1] - tooth2.centroid[1])
                
                if x_diff < 50 and y_diff > 30 and y_diff < 150:  # Aligned horizontally but separated vertically
                    # Determine which is on top
                    if tooth1.centroid[1] < tooth2.centroid[1]:
                        top, bottom = tooth1, tooth2
                    else:
                        top, bottom = tooth2, tooth1
                    
                    # In the upper jaw, primary molar should be further from the jaw edge
                    if "upper" in position:
                        if top.area < bottom.area or top.area/bottom.area < 1.2:
                            top.type = "premolar"
                            bottom.type = "primary_molar"
                    # In the lower jaw, primary molar should also be further from the jaw edge
                    else:
                        if bottom.area < top.area or bottom.area/top.area < 1.2:
                            bottom.type = "premolar"
                            top.type = "primary_molar"


def ensure_tooth_pairs(teeth, image):
    """Ensure that we identify both primary molars and premolars in each quadrant.
    
    If a quadrant has only one type of tooth detected, adjusts classification
    to create potential tooth pairs.
    
    Args:
        teeth (list): List of Tooth objects
        image (numpy.ndarray): Input image
        
    Returns:
        None: Updates the teeth list in-place
    """
    # Group teeth by quadrant
    quadrants = {
        "upper_left": [],
        "upper_right": [],
        "lower_left": [],
        "lower_right": []
    }
    
    for tooth in teeth:
        quadrants[tooth.position].append(tooth)
    
    # For each quadrant, check if we have both types of teeth
    for position, quadrant_teeth in quadrants.items():
        primary_molars = [t for t in quadrant_teeth if t.type == "primary_molar"]
        premolars = [t for t in quadrant_teeth if t.type == "premolar"]
        
        # If we have neither, try to classify at least one of each
        if not primary_molars and not premolars and len(quadrant_teeth) >= 2:
            # Sort by area, largest first
            quadrant_teeth.sort(key=lambda t: t.area, reverse=True)
            
            # Classify the largest as primary molar, the second largest as premolar
            if len(quadrant_teeth) > 0:
                quadrant_teeth[0].type = "primary_molar"
            if len(quadrant_teeth) > 1:
                quadrant_teeth[1].type = "premolar"
        
        # If we have primary molars but no premolars, try to identify potential premolars
        elif primary_molars and not premolars and len(quadrant_teeth) > len(primary_molars):
            # Sort remaining teeth by area
            other_teeth = [t for t in quadrant_teeth if t.type == "other"]
            other_teeth.sort(key=lambda t: t.area)
            
            # Look for teeth that might be premolars
            for tooth in other_teeth:
                # For each primary molar, check if there's a potential premolar nearby
                for primary in primary_molars:
                    x_diff = abs(primary.centroid[0] - tooth.centroid[0])
                    y_diff = abs(primary.centroid[1] - tooth.centroid[1])
                    
                    if x_diff < 75 and 30 < y_diff < 150:
                        tooth.type = "premolar"
                        break
            
            # If still no premolars, classify the smallest "other" tooth as a premolar
            if not [t for t in quadrant_teeth if t.type == "premolar"] and other_teeth:
                other_teeth[0].type = "premolar"
        
        # If we have premolars but no primary molars, try to identify potential primary molars
        elif premolars and not primary_molars and len(quadrant_teeth) > len(premolars):
            # Sort remaining teeth by area, largest first
            other_teeth = [t for t in quadrant_teeth if t.type == "other"]
            other_teeth.sort(key=lambda t: t.area, reverse=True)
            
            # Look for teeth that might be primary molars
            for tooth in other_teeth:
                # For each premolar, check if there's a potential primary molar nearby
                for premolar in premolars:
                    x_diff = abs(premolar.centroid[0] - tooth.centroid[0])
                    y_diff = abs(premolar.centroid[1] - tooth.centroid[1])
                    
                    if x_diff < 75 and 30 < y_diff < 150:
                        tooth.type = "primary_molar"
                        break
            
            # If still no primary molars, classify the largest "other" tooth as a primary molar
            if not [t for t in quadrant_teeth if t.type == "primary_molar"] and other_teeth:
                other_teeth[0].type = "primary_molar"
