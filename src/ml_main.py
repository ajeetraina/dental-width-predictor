#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main module for Dental Width Predictor using machine learning models.

This module serves as the entry point for the ML-based application,
orchestrating the workflow from image loading to measurement visualization.
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
from src.detection.ml_tooth_detection import detect_teeth_ml, filter_teeth_by_location, merge_overlapping_contours
from src.detection.ml_tooth_classification import classify_teeth_ml
from src.measurement.width_measurement import measure_tooth_width
from src.utils.visualization import visualize_measurements
from src.utils.calibration import calibrate_image


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor (ML-based)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the dental radiograph image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--calibration', type=float, default=None,
                        help='Calibration factor (mm/pixel)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    parser.add_argument('--fallback', action='store_true',
                        help='Fall back to traditional methods if ML fails')
    return parser.parse_args()


def main():
    """Main function to process dental radiographs using ML models."""
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
    
    # Detect teeth using ML
    try:
        teeth_contours, additional_data = detect_teeth_ml(processed_image, args.model_dir)
        
        # Save segmentation mask if in debug mode
        if args.debug and additional_data.get("method") == "ml" and args.output:
            segmentation_mask = additional_data.get("segmentation_mask")
            if segmentation_mask is not None:
                # Create a color visualization of the segmentation
                color_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
                color_mask[segmentation_mask == 1] = [0, 255, 0]  # Green for primary molars
                color_mask[segmentation_mask == 2] = [0, 0, 255]  # Red for premolars
                
                # Resize to match original image
                resized_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
                
                # Blend with original image
                alpha = 0.3
                blended = cv2.addWeighted(image, 1 - alpha, resized_mask, alpha, 0)
                
                # Save the visualization
                mask_path = os.path.splitext(args.output)[0] + "_segmentation.jpg"
                cv2.imwrite(mask_path, blended)
                print(f"Segmentation mask saved to: {mask_path}")
    
    except Exception as e:
        print(f"Error in ML tooth detection: {str(e)}")
        if args.fallback:
            print("Falling back to traditional methods.")
            from src.detection.tooth_detection import detect_teeth
            teeth_contours = detect_teeth(processed_image)
            additional_data = {"method": "traditional"}
        else:
            raise
    
    # Filter and merge contours
    teeth_contours = filter_teeth_by_location(processed_image, teeth_contours)
    teeth_contours = merge_overlapping_contours(teeth_contours)
    
    # Classify teeth
    try:
        classified_teeth = classify_teeth_ml(
            processed_image, teeth_contours, args.model_dir, additional_data
        )
    except Exception as e:
        print(f"Error in ML tooth classification: {str(e)}")
        if args.fallback:
            print("Falling back to traditional classification.")
            from src.detection.tooth_classification import classify_teeth
            classified_teeth = classify_teeth(processed_image, teeth_contours)
        else:
            raise
    
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
        cv2.imshow("Dental Width Measurements (ML-based)", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()