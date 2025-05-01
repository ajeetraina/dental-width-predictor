#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch processing module for Dental Width Predictor.

This module implements functions to process multiple dental radiographs,
save the results, and analyze the measurements.
"""

import os
import sys
import json
import csv
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_processing import preprocess_image
from src.detection.tooth_detection import detect_teeth, filter_teeth_by_location, merge_overlapping_contours
from src.detection.tooth_classification import classify_teeth
from src.measurement.width_measurement import measure_tooth_width
from src.utils.visualization import visualize_measurements
from src.utils.calibration import calibrate_image


def process_single_image(image_path, output_dir=None, calibration_factor=None, debug=False):
    """Process a single dental radiograph image.
    
    Args:
        image_path (str): Path to the dental radiograph image
        output_dir (str): Directory to save the results
        calibration_factor (float): Calibration factor (mm/pixel)
        debug (bool): Enable debug mode with additional visualizations
        
    Returns:
        dict: Measurement results
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None
    
    # Get the base filename without extension
    base_name = Path(image_path).stem
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Save preprocessed image if in debug mode
    if debug and output_dir:
        preprocessed_path = os.path.join(output_dir, f"{base_name}_preprocessed.jpg")
        cv2.imwrite(preprocessed_path, processed_image)
    
    # Calibrate the image
    if calibration_factor is None:
        calibration_factor = calibrate_image(processed_image)
    
    # Detect teeth
    contours = detect_teeth(processed_image)
    filtered_contours = filter_teeth_by_location(processed_image, contours)
    merged_contours = merge_overlapping_contours(filtered_contours)
    
    # Save contour detection image if in debug mode
    if debug and output_dir:
        detection_image = processed_image.copy()
        if len(detection_image.shape) == 2:  # If grayscale
            detection_image = cv2.cvtColor(detection_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(detection_image, merged_contours, -1, (0, 255, 0), 2)
        detection_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
        cv2.imwrite(detection_path, detection_image)
    
    # Classify teeth
    classified_teeth = classify_teeth(processed_image, merged_contours)
    
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
    
    # Print measurements
    print(f"\nImage: {image_path}")
    print("Width Measurements:")
    print("=====================")
    for i, result in enumerate(results):
        print(f"Tooth Pair {i+1}:")
        print(f"  Primary Molar Width: {result['primary_molar']['width']:.2f} mm")
        print(f"  Premolar Width: {result['premolar']['width']:.2f} mm")
        print(f"  Width Difference: {result['width_difference']:.2f} mm")
    
    # Create a structured result object for JSON output
    result_data = {
        "image": image_path,
        "processed_date": datetime.now().isoformat(),
        "calibration_factor": calibration_factor,
        "tooth_pairs": [
            {
                "pair_index": i,
                "primary_molar_width": result['primary_molar']['width'],
                "premolar_width": result['premolar']['width'],
                "width_difference": result['width_difference'],
                "position": result['primary_molar']['tooth'].position
            }
            for i, result in enumerate(results)
        ],
        "summary": {
            "total_pairs": len(results),
            "average_difference": sum(r['width_difference'] for r in results) / len(results) if results else 0
        }
    }
    
    # Save results to JSON if output directory is specified
    if output_dir:
        json_path = os.path.join(output_dir, f"{base_name}_measurements.json")
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    # Visualize results
    visualization = visualize_measurements(image, results)
    
    # Save visualization
    if output_dir:
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_path, visualization)
    
    return result_data


def batch_process_directory(input_dir, output_dir=None, calibration_factor=None, debug=False):
    """Process all dental radiograph images in a directory.
    
    Args:
        input_dir (str): Directory containing dental radiograph images
        output_dir (str): Directory to save the results
        calibration_factor (float): Calibration factor (mm/pixel)
        debug (bool): Enable debug mode with additional visualizations
        
    Returns:
        list: List of measurement results
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return None
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_dir).glob(f"*{ext}")))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return None
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
        result = process_single_image(
            str(image_path),
            output_dir,
            calibration_factor,
            debug
        )
        
        if result:
            all_results.append(result)
    
    # Generate summary CSV
    if output_dir and all_results:
        csv_path = os.path.join(output_dir, "measurements_summary.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Image", "Pair", "Position", "Primary Molar Width (mm)",
                "Premolar Width (mm)", "Width Difference (mm)"
            ])
            
            for result in all_results:
                image_name = Path(result["image"]).name
                
                for pair in result["tooth_pairs"]:
                    writer.writerow([
                        image_name,
                        pair["pair_index"] + 1,
                        pair["position"],
                        pair["primary_molar_width"],
                        pair["premolar_width"],
                        pair["width_difference"]
                    ])
        
        print(f"\nSummary saved to {csv_path}")
    
    return all_results


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor Batch Processing')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default='results',
                        help='Path to save the output visualizations and measurements')
    parser.add_argument('--calibration', type=float, default=None,
                        help='Calibration factor (mm/pixel)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    
    args = parser.parse_args()
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        batch_process_directory(
            args.input,
            args.output,
            args.calibration,
            args.debug
        )
    else:
        process_single_image(
            args.input,
            args.output,
            args.calibration,
            args.debug
        )


if __name__ == "__main__":
    main()
