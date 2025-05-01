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
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML-based detection if available, otherwise fall back to traditional
try:
    from src.detection.ml_tooth_detection import detect_teeth_ml as detect_teeth
    from src.detection.ml_tooth_classification import classify_teeth_ml as classify_teeth
    USE_ML = True
except ImportError:
    from src.preprocessing.image_processing import preprocess_image, enhance_teeth_boundaries
    from src.detection.tooth_detection import detect_teeth, filter_teeth_by_location, merge_overlapping_contours, refine_tooth_contours
    from src.detection.tooth_classification import classify_teeth
    USE_ML = False

from src.measurement.width_measurement import measure_tooth_width, analyze_width_ratios
from src.utils.visualization import visualize_measurements
from src.utils.calibration import calibrate_image


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the dental radiograph image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization')
    parser.add_argument('--result', type=str, default=None,
                        help='Path to save the measurement results as JSON')
    parser.add_argument('--calibration', type=float, default=0.1,
                        help='Calibration factor (mm/pixel), default=0.1')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    parser.add_argument('--method', choices=['ml', 'traditional'], default=None,
                        help='Force a specific detection method (ml or traditional)')
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
    
    # Determine method to use
    use_ml = USE_ML
    if args.method == 'ml':
        use_ml = True
    elif args.method == 'traditional':
        use_ml = False
    
    # Preprocess image
    if use_ml:
        # For ML-based approach, simple preprocessing
        from src.preprocessing.image_processing import preprocess_image
        processed_image = preprocess_image(image)
    else:
        # For traditional approach, use enhanced preprocessing
        processed_image = preprocess_image(image)
        processed_image = enhance_teeth_boundaries(processed_image)
    
    # Calibrate image if necessary
    calibration_factor = args.calibration
    
    # Save intermediate results for debugging
    if args.debug:
        debug_dir = os.path.join(os.path.dirname(args.output) if args.output else ".", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "1_preprocessed.jpg"), processed_image)
    
    # Detect teeth
    if use_ml:
        teeth_contours, detection_data = detect_teeth(processed_image, model_dir="models")
        if args.debug and "segmentation_mask" in detection_data:
            # Save segmentation mask
            cv2.imwrite(os.path.join(debug_dir, "2_segmentation_mask.jpg"), 
                       detection_data["segmentation_mask"] * 80)  # Scale for visibility
    else:
        # Traditional approach with improved pipeline
        teeth_contours = detect_teeth(processed_image)
        
        # Filter teeth by location in dental arch
        teeth_contours = filter_teeth_by_location(processed_image, teeth_contours)
        
        # Merge overlapping contours
        teeth_contours = merge_overlapping_contours(teeth_contours)
        
        # Refine contours
        teeth_contours = refine_tooth_contours(processed_image, teeth_contours)
        
        detection_data = {"method": "traditional"}
    
    # Save teeth contours for debugging
    if args.debug:
        contour_image = image.copy()
        cv2.drawContours(contour_image, teeth_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "3_detected_teeth.jpg"), contour_image)
    
    # Classify teeth (identify primary second molars and second premolars)
    if use_ml:
        classified_teeth = classify_teeth(processed_image, teeth_contours, model_dir="models", additional_data=detection_data)
    else:
        classified_teeth = classify_teeth(processed_image, teeth_contours)
    
    # Save classified teeth for debugging
    if args.debug:
        classified_image = image.copy()
        for tooth in classified_teeth:
            color = (0, 255, 0) if tooth.type == "primary_molar" else (0, 0, 255) if tooth.type == "premolar" else (255, 0, 0)
            cv2.drawContours(classified_image, [tooth.contour], -1, color, 2)
            cv2.putText(classified_image, tooth.type, (tooth.centroid[0], tooth.centroid[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_dir, "4_classified_teeth.jpg"), classified_image)
    
    # Measure tooth widths
    measurements = measure_tooth_width(processed_image, classified_teeth, calibration_factor)
    
    # Analyze results
    analysis = analyze_width_ratios(measurements)
    
    # Prepare results data
    results_data = {
        "image": os.path.basename(image_path),
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
    
    # Print results
    print(f"\nImage: {image_path}")
    print("Width Measurements:")
    print("=====================")
    
    if not measurements:
        print("No valid tooth pairs found for measurement.")
    else:
        for i, result in enumerate(measurements):
            primary_width = result["primary_molar"]["measurement"]["width"]
            premolar_width = result["premolar"]["measurement"]["width"]
            difference = result["width_difference"]
            
            print(f"Tooth Pair {i+1} ({result['primary_molar']['position']}):")
            print(f"  Primary Molar Width: {primary_width:.2f} mm")
            print(f"  Premolar Width: {premolar_width:.2f} mm")
            print(f"  Width Difference: {difference:.2f} mm")
        
        print(f"\nSummary:")
        print(f"  Total tooth pairs: {results_data['summary']['total_pairs']}")
        print(f"  Average width difference: {results_data['summary']['average_difference']:.2f} mm")
        print(f"  Average primary/premolar ratio: {results_data['analysis']['average_ratio']:.2f}")
    
    # Save results to JSON if specified
    if args.result or (args.output and not args.result):
        result_path = args.result
        if not result_path:
            # If output path is specified but not result, use output path with .json extension
            output_dir = os.path.dirname(args.output)
            base_name = os.path.splitext(os.path.basename(args.output))[0]
            result_path = os.path.join(output_dir, f"{base_name}_measurements.json")
        
        os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nMeasurement results saved to: {result_path}")
    
    # Visualize results
    visualization = visualize_measurements(image, measurements)
    
    # Save or show output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        cv2.imwrite(args.output, visualization)
        print(f"\nVisualization saved to: {args.output}")
    else:
        # Try to show the image if possible
        try:
            cv2.imshow("Dental Width Measurements", visualization)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image (likely running in headless mode).")
            # Save to a default location
            output_path = os.path.splitext(image_path)[0] + "_measurements.jpg"
            cv2.imwrite(output_path, visualization)
            print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
