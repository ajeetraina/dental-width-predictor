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
    
    for contour in contours:
        # Calculate contour properties
        M = cv2.moments(contour)
        if M["m00"] == 0:  # Skip if contour area is zero
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine tooth position (quadrant)
        if cx < midline_x and cy < midline_y:
            position = "upper_left"
        elif cx >= midline_x and cy < midline_y:
            position = "upper_right"
        elif cx < midline_x and cy >= midline_y:
            position = "lower_left"
        else:
            position = "lower_right"
        
        # Initial classification based on size and location
        # This is a simplistic approach and would need refinement with actual data
        if area > 2000:  # Larger teeth are likely molars
            tooth_type = "primary_molar"
        elif area > 1000:  # Medium teeth could be premolars
            tooth_type = "premolar"
        else:  # Smaller teeth could be incisors or other
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
    
    # Further refine classification using relative positions
    refine_classification(classified_teeth, image)
    
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
            
        # Sort teeth by x-coordinate
        if "left" in position:
            quadrant_teeth.sort(key=lambda t: t.centroid[0], reverse=True)  # From midline outward
        else:
            quadrant_teeth.sort(key=lambda t: t.centroid[0])  # From midline outward
        
        # Find vertically aligned teeth (potential primary molar-premolar pairs)
        for i, tooth1 in enumerate(quadrant_teeth):
            for j, tooth2 in enumerate(quadrant_teeth):
                if i == j:
                    continue
                    
                # Check if they're roughly vertically aligned
                x_diff = abs(tooth1.centroid[0] - tooth2.centroid[0])
                y_diff = abs(tooth1.centroid[1] - tooth2.centroid[1])
                
                if x_diff < 50 and y_diff > 50:  # Aligned horizontally but separated vertically
                    # Determine which is on top
                    if tooth1.centroid[1] < tooth2.centroid[1]:
                        top, bottom = tooth1, tooth2
                    else:
                        top, bottom = tooth2, tooth1
                    
                    # In the upper jaw, primary molar is on bottom
                    if "upper" in position:
                        if top.area < bottom.area:
                            top.type = "premolar"
                            bottom.type = "primary_molar"
                    # In the lower jaw, primary molar is on top
                    else:
                        if top.area > bottom.area:
                            top.type = "primary_molar"
                            bottom.type = "premolar"
