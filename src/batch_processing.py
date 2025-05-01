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
import traceback
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML-based detection if available, otherwise fall back to traditional
try:
    from src.detection.ml_tooth_detection import detect_teeth_ml as detect_ml
    from src.detection.ml_tooth_classification import classify_teeth_ml as classify_ml
    USE_ML = True
except ImportError:
    USE_ML = False

from src.preprocessing.image_processing import preprocess_image, enhance_teeth_boundaries, enhance_edges
from src.detection.tooth_detection import detect_teeth, filter_teeth_by_location, merge_overlapping_contours, refine_tooth_contours
from src.detection.tooth_classification import classify_teeth, ensure_tooth_pairs
from src.measurement.width_measurement import measure_tooth_width, analyze_width_ratios
from src.utils.visualization import visualize_measurements
from src.utils.calibration import calibrate_image


def process_single_image(image_path, output_dir=None, calibration_factor=0.1, debug=False, method=None):
    """Process a single dental radiograph image.
    
    Args:
        image_path (str): Path to the dental radiograph image
        output_dir (str): Directory to save the results
        calibration_factor (float): Calibration factor (mm/pixel)
        debug (bool): Enable debug mode with additional visualizations
        method (str): Force a specific detection method ('ml' or 'traditional')
        
    Returns:
        dict: Measurement results
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
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
        
        # Determine method to use
        use_ml = USE_ML
        if method == 'ml':
            use_ml = True
        elif method == 'traditional':
            use_ml = False
        
        # Create debug directory if needed
        debug_dir = None
        if debug and output_dir:
            debug_dir = os.path.join(output_dir, f"{base_name}_debug")
            os.makedirs(debug_dir, exist_ok=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Save preprocessed image if in debug mode
        if debug_dir:
            preprocessed_path = os.path.join(debug_dir, "1_preprocessed.jpg")
            cv2.imwrite(preprocessed_path, processed_image)
        
        # Enhance tooth boundaries for traditional approach
        if not use_ml:
            enhanced_image = enhance_teeth_boundaries(processed_image)
            if debug_dir:
                enhanced_path = os.path.join(debug_dir, "2_enhanced.jpg")
                cv2.imwrite(enhanced_path, enhanced_image)
            processed_image = enhanced_image
        
        # Detect teeth
        detection_data = {}
        if use_ml:
            try:
                contours, detection_data = detect_ml(processed_image, model_dir="models")
            except Exception as e:
                print(f"ML detection failed, falling back to traditional: {str(e)}")
                use_ml = False
        
        if not use_ml:
            # Traditional approach with improved steps
            contours = detect_teeth(processed_image)
            contours = filter_teeth_by_location(processed_image, contours)
            contours = merge_overlapping_contours(contours)
            contours = refine_tooth_contours(processed_image, contours)
            detection_data = {"method": "traditional"}
        
        # Save contour detection image if in debug mode
        if debug_dir:
            detection_image = image.copy()
            cv2.drawContours(detection_image, contours, -1, (0, 255, 0), 2)
            detection_path = os.path.join(debug_dir, "3_detected_teeth.jpg")
            cv2.imwrite(detection_path, detection_image)
        
        # Classify teeth
        if use_ml:
            try:
                classified_teeth = classify_ml(processed_image, contours, model_dir="models", additional_data=detection_data)
            except Exception as e:
                print(f"ML classification failed, falling back to traditional: {str(e)}")
                classified_teeth = classify_teeth(processed_image, contours)
        else:
            classified_teeth = classify_teeth(processed_image, contours)
            # Ensure we have tooth pairs
            ensure_tooth_pairs(classified_teeth, processed_image)
        
        # Save classified teeth image if in debug mode
        if debug_dir:
            classified_image = image.copy()
            for tooth in classified_teeth:
                color = (0, 255, 0) if tooth.type == "primary_molar" else \
                        (0, 0, 255) if tooth.type == "premolar" else (255, 0, 0)
                cv2.drawContours(classified_image, [tooth.contour], -1, color, 2)
                cv2.putText(classified_image, tooth.type[:5], (tooth.centroid[0], tooth.centroid[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            classified_path = os.path.join(debug_dir, "4_classified_teeth.jpg")
            cv2.imwrite(classified_path, classified_image)
        
        # Measure tooth widths
        measurements = measure_tooth_width(processed_image, classified_teeth, calibration_factor)
        
        # Analyze results
        analysis = analyze_width_ratios(measurements)
        
        # Create a structured result object for JSON output
        result_data = {
            "image": str(image_path),
            "processed_date": datetime.now().isoformat(),
            "calibration_factor": calibration_factor,
            "tooth_pairs": measurements,
            "summary": {
                "total_pairs": len(measurements),
                "average_difference": np.mean([pair["width_difference"] for pair in measurements]) if measurements else 0
            },
            "analysis": {
                "average_ratio": float(analysis["average_ratio"]),
                "std_deviation": float(analysis["std_deviation"]),
                "valid_pairs": analysis["valid_pairs"]
            }
        }
        
        # Save results to JSON if output directory is specified
        if output_dir:
            json_path = os.path.join(output_dir, f"{base_name}_measurements.json")
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
        
        # Visualize results
        visualization = visualize_measurements(image, measurements)
        
        # Save visualization
        if output_dir:
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(vis_path, visualization)
        
        return result_data
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None


def batch_process_directory(input_dir, output_dir=None, calibration_factor=0.1, debug=False, method=None):
    """Process all dental radiograph images in a directory.
    
    Args:
        input_dir (str): Directory containing dental radiograph images
        output_dir (str): Directory to save the results
        calibration_factor (float): Calibration factor (mm/pixel)
        debug (bool): Enable debug mode with additional visualizations
        method (str): Force a specific detection method ('ml' or 'traditional')
        
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
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.lower()}")))
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
    
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
            debug,
            method
        )
        
        if result:
            all_results.append(result)
    
    # Generate summary CSV
    if output_dir and all_results:
        csv_path = os.path.join(output_dir, "measurements_summary.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Image", "Primary Molar Width (mm)",
                "Premolar Width (mm)", "Width Difference (mm)", "Position"
            ])
            
            for result in all_results:
                image_name = Path(result["image"]).name
                
                for pair in result["tooth_pairs"]:
                    try:
                        primary_width = pair["primary_molar"]["measurement"]["width"]
                        premolar_width = pair["premolar"]["measurement"]["width"]
                        difference = pair["width_difference"]
                        position = pair["primary_molar"]["position"]
                        
                        writer.writerow([
                            image_name,
                            f"{primary_width:.2f}",
                            f"{premolar_width:.2f}",
                            f"{difference:.2f}",
                            position
                        ])
                    except KeyError as e:
                        print(f"Warning: Could not extract complete data for {image_name}: {e}")
                        continue
        
        print(f"\nSummary saved to {csv_path}")
    
    return all_results


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor Batch Processing')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default='results',
                        help='Path to save the output visualizations and measurements')
    parser.add_argument('--calibration', type=float, default=0.1,
                        help='Calibration factor (mm/pixel), default=0.1')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    parser.add_argument('--method', choices=['ml', 'traditional'], default=None,
                        help='Force a specific detection method (ml or traditional)')
    
    args = parser.parse_args()
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        results = batch_process_directory(
            args.input,
            args.output,
            args.calibration,
            args.debug,
            args.method
        )
    else:
        results = process_single_image(
            args.input,
            args.output,
            args.calibration,
            args.debug,
            args.method
        )
    
    if not results:
        print("\nNo valid measurement results were produced.")
    else:
        print("\nProcessing complete!")


if __name__ == "__main__":
    main()
