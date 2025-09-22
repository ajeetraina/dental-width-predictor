#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced batch processing module for Dental Width Predictor with improved debugging.

This module implements functions to process multiple dental radiographs with
better error handling, debugging, and fallback mechanisms.
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
from typing import List, Dict, Optional, Tuple

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML-based detection if available, otherwise fall back to traditional
try:
    from src.detection.ml_tooth_detection import detect_teeth_ml as detect_ml
    from src.detection.ml_tooth_classification import classify_teeth_ml as classify_ml
    USE_ML = True
    print("✓ ML models available")
except ImportError:
    USE_ML = False
    print("⚠ ML models not available, using traditional methods")

from src.preprocessing.image_processing import preprocess_image, enhance_teeth_boundaries, enhance_edges
from src.detection.tooth_detection import detect_teeth, filter_teeth_by_location, merge_overlapping_contours, refine_tooth_contours
from src.detection.tooth_classification import classify_teeth, ensure_tooth_pairs
from src.measurement.width_measurement import measure_tooth_width, analyze_width_ratios
from src.utils.visualization import visualize_measurements
from src.utils.calibration import calibrate_image


def enhanced_detect_teeth(image, debug_dir=None):
    """Enhanced tooth detection with multiple parameter sets and debugging.
    
    Args:
        image (numpy.ndarray): Preprocessed radiograph image
        debug_dir (str): Directory to save debug images
        
    Returns:
        tuple: (contours, detection_info)
    """
    detection_info = {"method": "enhanced_traditional", "attempts": []}
    all_contours = []
    
    # Try multiple parameter sets for edge detection
    parameter_sets = [
        {"sigma1": 1.5, "sigma2": 2.5, "area_min": 200, "area_max": 25000},
        {"sigma1": 1.0, "sigma2": 3.0, "area_min": 150, "area_max": 30000},
        {"sigma1": 2.0, "sigma2": 3.5, "area_min": 100, "area_max": 35000},
        {"sigma1": 0.8, "sigma2": 4.0, "area_min": 250, "area_max": 20000}
    ]
    
    for i, params in enumerate(parameter_sets):
        try:
            # Apply different preprocessing for each attempt
            if i == 0:
                processed = image
            elif i == 1:
                processed = enhance_teeth_boundaries(image)
            elif i == 2:
                # More aggressive enhancement
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                processed = clahe.apply(image)
            else:
                # Gaussian blur to smooth noise
                processed = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Edge detection with current parameters
            from skimage import feature, exposure
            p2, p98 = np.percentile(processed, (2, 98))
            image_enhanced = exposure.rescale_intensity(processed, in_range=(p2, p98))
            
            edges1 = feature.canny(image_enhanced, sigma=params["sigma1"])
            edges2 = feature.canny(image_enhanced, sigma=params["sigma2"])
            edges = np.logical_or(edges1, edges2)
            
            # Morphological operations
            from skimage import morphology, measure
            dilated = morphology.dilation(edges, morphology.disk(2))
            filled = morphology.remove_small_holes(dilated, area_threshold=50)
            closed = morphology.closing(filled, morphology.disk(3))
            
            # Label and filter components
            labeled = measure.label(closed)
            props = measure.regionprops(labeled)
            
            valid_contours = []
            for prop in props:
                if (params["area_min"] <= prop.area <= params["area_max"] and 
                    prop.eccentricity >= 0.05 and prop.solidity >= 0.3):  # More permissive thresholds
                    
                    binary = (labeled == prop.label).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    
                    for cnt in contours:
                        if cv2.contourArea(cnt) > params["area_min"]:
                            valid_contours.append(cnt)
            
            all_contours.extend(valid_contours)
            
            attempt_info = {
                "parameters": params,
                "contours_found": len(valid_contours),
                "total_components": len(props)
            }
            detection_info["attempts"].append(attempt_info)
            
            # Save debug image for this attempt
            if debug_dir and valid_contours:
                debug_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 2)
                debug_path = os.path.join(debug_dir, f"detection_attempt_{i+1}.jpg")
                cv2.imwrite(debug_path, debug_img)
                
        except Exception as e:
            detection_info["attempts"].append({"error": str(e), "parameters": params})
    
    # Remove duplicate contours and filter by location
    if all_contours:
        all_contours = filter_teeth_by_location(image, all_contours)
        all_contours = merge_overlapping_contours(all_contours, threshold=0.2)
    
    detection_info["final_count"] = len(all_contours)
    return all_contours, detection_info


def enhanced_classify_teeth(image, contours, debug_dir=None):
    """Enhanced tooth classification with better debugging and fallback strategies.
    
    Args:
        image (numpy.ndarray): Input image
        contours (list): List of tooth contours
        debug_dir (str): Directory to save debug images
        
    Returns:
        tuple: (classified_teeth, classification_info)
    """
    classification_info = {"method": "enhanced", "original_count": len(contours)}
    
    try:
        # Use the original classification
        classified_teeth = classify_teeth(image, contours)
        
        # Apply more aggressive pairing strategies
        ensure_tooth_pairs(classified_teeth, image)
        
        # If still no pairs, try alternative classification
        primary_molars = [t for t in classified_teeth if t.type == "primary_molar"]
        premolars = [t for t in classified_teeth if t.type == "premolar"]
        
        if len(primary_molars) == 0 or len(premolars) == 0:
            # Apply size-based classification as fallback
            areas = [cv2.contourArea(t.contour) for t in classified_teeth]
            if areas:
                median_area = np.median(areas)
                
                for tooth in classified_teeth:
                    tooth_area = cv2.contourArea(tooth.contour)
                    
                    if tooth_area > median_area * 1.2:
                        tooth.type = "primary_molar"
                    elif tooth_area > median_area * 0.8:
                        tooth.type = "premolar"
                    else:
                        tooth.type = "other"
        
        # Aggressive fallback: if still no pairs, force alternating classification
        primary_molars = [t for t in classified_teeth if t.type == "primary_molar"]
        premolars = [t for t in classified_teeth if t.type == "premolar"]
        
        if (len(primary_molars) == 0 or len(premolars) == 0) and classified_teeth:
            # Sort by area and alternate classifications
            classified_teeth.sort(key=lambda t: cv2.contourArea(t.contour), reverse=True)
            for i, tooth in enumerate(classified_teeth):
                tooth.type = "primary_molar" if i % 2 == 0 else "premolar"
        
        # Count final classifications
        final_counts = {}
        for tooth in classified_teeth:
            final_counts[tooth.type] = final_counts.get(tooth.type, 0) + 1
        
        classification_info["classifications"] = final_counts
        
        # Save classification debug image
        if debug_dir and classified_teeth:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            colors = {"primary_molar": (0, 255, 0), "premolar": (0, 0, 255), "other": (255, 0, 0)}
            
            for tooth in classified_teeth:
                color = colors.get(tooth.type, (128, 128, 128))
                cv2.drawContours(debug_img, [tooth.contour], -1, color, 2)
                
                # Add text label
                text = f"{tooth.type[:8]}"
                cv2.putText(debug_img, text, tooth.centroid, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            debug_path = os.path.join(debug_dir, "classification_result.jpg")
            cv2.imwrite(debug_path, debug_img)
        
        return classified_teeth, classification_info
        
    except Exception as e:
        classification_info["error"] = str(e)
        return [], classification_info


def process_single_image_enhanced(image_path, output_dir=None, calibration_factor=0.1, 
                                 debug=False, method=None, save_failed=True):
    """Enhanced version of single image processing with comprehensive debugging.
    
    Args:
        image_path (str): Path to the dental radiograph image
        output_dir (str): Directory to save the results
        calibration_factor (float): Calibration factor (mm/pixel)
        debug (bool): Enable debug mode with additional visualizations
        method (str): Force a specific detection method ('ml' or 'traditional')
        save_failed (bool): Save results even for failed cases
        
    Returns:
        dict: Detailed processing results
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        return {"error": f"Image file not found: {image_path}", "success": False}
    
    processing_log = {
        "image": str(image_path),
        "processed_date": datetime.now().isoformat(),
        "success": False,
        "steps": {},
        "errors": []
    }
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            processing_log["errors"].append(f"Failed to load image: {image_path}")
            return processing_log
        
        processing_log["steps"]["load"] = {"success": True, "shape": image.shape}
        
        # Get the base filename without extension
        base_name = Path(image_path).stem
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create debug directory if needed
        debug_dir = None
        if debug and output_dir:
            debug_dir = os.path.join(output_dir, f"{base_name}_debug")
            os.makedirs(debug_dir, exist_ok=True)
        
        # Step 1: Preprocess the image
        try:
            processed_image = preprocess_image(image)
            processing_log["steps"]["preprocess"] = {"success": True}
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "1_preprocessed.jpg"), processed_image)
        except Exception as e:
            processing_log["errors"].append(f"Preprocessing failed: {str(e)}")
            processing_log["steps"]["preprocess"] = {"success": False, "error": str(e)}
            return processing_log
        
        # Step 2: Detect teeth
        try:
            if USE_ML and method != 'traditional':
                try:
                    contours, detection_data = detect_ml(processed_image, model_dir="models")
                    processing_log["steps"]["detection"] = {"success": True, "method": "ML", "data": detection_data}
                except Exception as e:
                    print(f"ML detection failed, falling back to enhanced traditional: {str(e)}")
                    contours, detection_data = enhanced_detect_teeth(processed_image, debug_dir)
                    processing_log["steps"]["detection"] = {"success": True, "method": "traditional_enhanced", "data": detection_data}
            else:
                contours, detection_data = enhanced_detect_teeth(processed_image, debug_dir)
                processing_log["steps"]["detection"] = {"success": True, "method": "traditional_enhanced", "data": detection_data}
            
            processing_log["contours_detected"] = len(contours)
            
            if not contours:
                processing_log["errors"].append("No tooth contours detected")
                if not save_failed:
                    return processing_log
            
        except Exception as e:
            processing_log["errors"].append(f"Tooth detection failed: {str(e)}")
            processing_log["steps"]["detection"] = {"success": False, "error": str(e)}
            if not save_failed:
                return processing_log
            contours = []
        
        # Step 3: Classify teeth
        try:
            if contours:
                classified_teeth, classification_info = enhanced_classify_teeth(processed_image, contours, debug_dir)
                processing_log["steps"]["classification"] = {"success": True, "data": classification_info}
                processing_log["teeth_classified"] = len(classified_teeth)
            else:
                classified_teeth = []
                processing_log["steps"]["classification"] = {"success": False, "error": "No contours to classify"}
            
        except Exception as e:
            processing_log["errors"].append(f"Tooth classification failed: {str(e)}")
            processing_log["steps"]["classification"] = {"success": False, "error": str(e)}
            classified_teeth = []
        
        # Step 4: Measure tooth widths
        measurements = []
        try:
            if classified_teeth:
                measurements = measure_tooth_width(processed_image, classified_teeth, calibration_factor)
                processing_log["steps"]["measurement"] = {"success": True, "pairs_found": len(measurements)}
                processing_log["measurements_found"] = len(measurements)
            else:
                processing_log["steps"]["measurement"] = {"success": False, "error": "No classified teeth available"}
            
        except Exception as e:
            processing_log["errors"].append(f"Width measurement failed: {str(e)}")
            processing_log["steps"]["measurement"] = {"success": False, "error": str(e)}
        
        # Step 5: Analyze results
        try:
            analysis = analyze_width_ratios(measurements)
            processing_log["steps"]["analysis"] = {"success": True}
        except Exception as e:
            processing_log["errors"].append(f"Analysis failed: {str(e)}")
            processing_log["steps"]["analysis"] = {"success": False, "error": str(e)}
            analysis = {"average_ratio": 0, "std_deviation": 0, "valid_pairs": 0}
        
        # Create result structure
        result_data = {
            "image": str(image_path),
            "processed_date": datetime.now().isoformat(),
            "calibration_factor": calibration_factor,
            "processing_log": processing_log,
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
        
        # Mark as successful if we found any measurements
        if measurements:
            processing_log["success"] = True
            result_data["processing_log"]["success"] = True
        
        # Save results to JSON if output directory is specified
        if output_dir:
            json_path = os.path.join(output_dir, f"{base_name}_measurements.json")
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
        
        # Create and save visualization
        try:
            if measurements or save_failed:
                visualization = visualize_measurements(image, measurements if measurements else [])
                
                if output_dir:
                    vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
                    cv2.imwrite(vis_path, visualization)
        except Exception as e:
            processing_log["errors"].append(f"Visualization failed: {str(e)}")
        
        return result_data
        
    except Exception as e:
        processing_log["errors"].append(f"Unexpected error: {str(e)}")
        processing_log["steps"]["unexpected_error"] = {"error": str(e), "traceback": traceback.format_exc()}
        if debug:
            traceback.print_exc()
        return processing_log


def batch_process_directory_enhanced(input_dir, output_dir=None, calibration_factor=0.1, 
                                    debug=False, method=None, max_failures=10):
    """Enhanced batch processing with better error handling and reporting.
    
    Args:
        input_dir (str): Directory containing dental radiograph images
        output_dir (str): Directory to save the results
        calibration_factor (float): Calibration factor (mm/pixel)
        debug (bool): Enable debug mode with additional visualizations
        method (str): Force a specific detection method ('ml' or 'traditional')
        max_failures (int): Maximum number of consecutive failures before stopping
        
    Returns:
        dict: Comprehensive processing results
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        return {"error": f"Input directory not found: {input_dir}"}
    
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
        return {"error": f"No images found in {input_dir}"}
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Initialize batch results
    batch_results = {
        "input_directory": input_dir,
        "output_directory": output_dir,
        "total_images": len(image_paths),
        "processed_images": 0,
        "successful_measurements": 0,
        "failed_images": 0,
        "results": [],
        "errors": [],
        "summary_stats": {},
        "processing_started": datetime.now().isoformat()
    }
    
    consecutive_failures = 0
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        result = process_single_image_enhanced(
            str(image_path),
            output_dir,
            calibration_factor,
            debug,
            method,
            save_failed=True
        )
        
        batch_results["results"].append(result)
        batch_results["processed_images"] += 1
        
        if result.get("success", False) and result.get("tooth_pairs", []):
            batch_results["successful_measurements"] += 1
            consecutive_failures = 0
            print(f"✓ Found {len(result['tooth_pairs'])} tooth pairs")
        else:
            batch_results["failed_images"] += 1
            consecutive_failures += 1
            errors = result.get("errors", result.get("processing_log", {}).get("errors", []))
            print(f"✗ Processing failed: {errors}")
            
            if consecutive_failures >= max_failures:
                print(f"\n⚠ Stopping after {consecutive_failures} consecutive failures")
                break
    
    batch_results["processing_completed"] = datetime.now().isoformat()
    
    # Generate enhanced CSV summary
    if output_dir and batch_results["successful_measurements"] > 0:
        generate_enhanced_csv_summary(batch_results, output_dir)
    
    # Generate batch processing report
    if output_dir:
        generate_batch_report(batch_results, output_dir)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total images processed: {batch_results['processed_images']}")
    print(f"Successful measurements: {batch_results['successful_measurements']}")
    print(f"Failed images: {batch_results['failed_images']}")
    print(f"Success rate: {batch_results['successful_measurements']/batch_results['processed_images']*100:.1f}%")
    
    return batch_results


def generate_enhanced_csv_summary(batch_results, output_dir):
    """Generate an enhanced CSV summary with additional metrics.
    
    Args:
        batch_results (dict): Batch processing results
        output_dir (str): Output directory
    """
    csv_path = os.path.join(output_dir, "measurements_summary_enhanced.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            "Image", "Success", "Primary Molar Width (mm)",
            "Premolar Width (mm)", "Width Difference (mm)", "Position",
            "Detection Method", "Contours Found", "Classifications",
            "Processing Errors"
        ])
        
        # Data rows
        for result in batch_results["results"]:
            if not isinstance(result, dict):
                continue
                
            image_name = Path(result.get("image", "unknown")).name
            success = result.get("success", False)
            
            processing_log = result.get("processing_log", {})
            detection_info = processing_log.get("steps", {}).get("detection", {})
            
            # Extract processing details
            detection_method = detection_info.get("method", "unknown")
            contours_found = result.get("contours_detected", 0)
            errors = "; ".join(result.get("errors", processing_log.get("errors", [])))
            
            tooth_pairs = result.get("tooth_pairs", [])
            
            if tooth_pairs and success:
                for pair in tooth_pairs:
                    try:
                        primary_width = pair["primary_molar"]["measurement"]["width"]
                        premolar_width = pair["premolar"]["measurement"]["width"]
                        difference = pair["width_difference"]
                        position = pair["primary_molar"]["position"]
                        
                        writer.writerow([
                            image_name, "Yes", f"{primary_width:.2f}",
                            f"{premolar_width:.2f}", f"{difference:.2f}", position,
                            detection_method, contours_found, 
                            f"PM:{len([t for t in tooth_pairs if 'primary_molar' in str(t)])}", errors
                        ])
                    except (KeyError, TypeError) as e:
                        continue
            else:
                # Write failed case
                writer.writerow([
                    image_name, "No", "", "", "", "",
                    detection_method, contours_found, "", errors
                ])
    
    print(f"Enhanced summary saved to {csv_path}")


def generate_batch_report(batch_results, output_dir):
    """Generate a detailed batch processing report.
    
    Args:
        batch_results (dict): Batch processing results
        output_dir (str): Output directory
    """
    report_path = os.path.join(output_dir, "batch_processing_report.json")
    
    with open(report_path, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"Batch processing report saved to {report_path}")


def main():
    """Enhanced main function with additional options."""
    parser = argparse.ArgumentParser(description='Enhanced Dental Width Predictor Batch Processing')
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
    parser.add_argument('--max-failures', type=int, default=10,
                        help='Maximum consecutive failures before stopping batch processing')
    
    args = parser.parse_args()
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        results = batch_process_directory_enhanced(
            args.input,
            args.output,
            args.calibration,
            args.debug,
            args.method,
            args.max_failures
        )
    else:
        results = process_single_image_enhanced(
            args.input,
            args.output,
            args.calibration,
            args.debug,
            args.method
        )
    
    if results.get("error"):
        print(f"\n❌ Processing failed: {results['error']}")
    else:
        print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
