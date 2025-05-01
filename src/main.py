#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main module for Dental Width Predictor.

This module serves as the entry point for the application, orchestrating the workflow
from image loading to measurement visualization.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_processing import preprocess_image
from src.detection.tooth_detection import detect_teeth
from src.detection.tooth_classification import classify_teeth
from src.measurement.width_measurement import measure_tooth_width
from src.utils.visualization import visualize_measurements
from src.utils.calibration import calibrate_image


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the dental radiograph image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization')
    parser.add_argument('--calibration', type=float, default=None,
                        help='Calibration factor (mm/pixel)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    return parser.parse_args()


def main():
    """Main function to process dental radiographs and measure tooth widths."""
    # Parse arguments
    args = parse_arguments()
    
    # Load image
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Calibrate image if necessary
    calibration_factor = args.calibration
    if calibration_factor is None:
        calibration_factor = calibrate_image(processed_image)
    
    # Detect teeth
    teeth_contours = detect_teeth(processed_image)
    
    # Classify teeth (identify primary second molars and second premolars)
    classified_teeth = classify_teeth(processed_image, teeth_contours)
    
    # Measure tooth widths
    measurements = measure_tooth_width(processed_image, classified_teeth, calibration_factor)
    
    # Calculate width differences
    results = []
    for primary, premolar in measurements:
        if primary and premolar:  # If both measurements exist
            primary_width = primary['width']
            premolar_width = premolar['width']
            difference = primary_width - premolar_width
            results.append({
                'primary_molar': primary,
                'premolar': premolar,
                'width_difference': difference
            })
    
    # Print results
    print("\nWidth Measurements:")
    print("=====================")
    for i, result in enumerate(results):
        print(f"Tooth Pair {i+1}:")
        print(f"  Primary Molar Width: {result['primary_molar']['width']:.2f} mm")
        print(f"  Premolar Width: {result['premolar']['width']:.2f} mm")
        print(f"  Width Difference: {result['width_difference']:.2f} mm")
    
    # Visualize results
    visualization = visualize_measurements(image, results)
    
    # Save or show output
    if args.output:
        cv2.imwrite(args.output, visualization)
        print(f"\nOutput saved to: {args.output}")
    else:
        cv2.imshow("Dental Width Measurements", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
