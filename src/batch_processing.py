#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch processing module for Dental Width Predictor.
Fixed version that uses our working AI models.
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

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our working models
from models.model import create_segmentation_model, extract_tooth_measurements

def create_tooth_pairs(measurements):
    """
    Group individual tooth measurements into primary molar - premolar pairs
    Same logic as in main.py
    """
    tooth_pairs = []
    
    # Group measurements by quadrant
    quadrants = {}
    for measurement in measurements:
        quadrant = measurement['position']
        if quadrant not in quadrants:
            quadrants[quadrant] = []
        quadrants[quadrant].append(measurement)
    
    # For each quadrant, try to pair primary molars with premolars
    for quadrant, teeth in quadrants.items():
        if len(teeth) >= 2:
            # Sort by area - larger teeth are likely primary molars
            teeth_sorted = sorted(teeth, key=lambda x: x['area'], reverse=True)
            
            # Pair largest with second largest (primary molar with premolar)
            primary_molar = teeth_sorted[0]
            premolar = teeth_sorted[1]
            
            # Only create pairs with reasonable measurements
            if (5 < primary_molar['mesiodistal_width_mm'] < 25 and 
                5 < premolar['mesiodistal_width_mm'] < 25):
                
                width_difference = primary_molar['mesiodistal_width_mm'] - premolar['mesiodistal_width_mm']
                
                tooth_pair = {
                    "quadrant": quadrant,
                    "primary_molar": {
                        "width_mm": primary_molar['mesiodistal_width_mm'],
                        "position": [primary_molar['center_x'], primary_molar['center_y']]
                    },
                    "premolar": {
                        "width_mm": premolar['mesiodistal_width_mm'], 
                        "position": [premolar['center_x'], premolar['center_y']]
                    },
                    "width_difference_mm": width_difference
                }
                
                tooth_pairs.append(tooth_pair)
    
    return tooth_pairs

def visualize_measurements(image, measurements, tooth_pairs):
    """
    Create visualization showing detected teeth and measurements
    Same as in main.py
    """
    visualization = image.copy()
    
    # Draw individual tooth measurements
    for measurement in measurements:
        x, y, w, h = measurement['bounding_box']
        
        # Draw bounding box
        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add measurement text
        text = f"{measurement['mesiodistal_width_mm']:.1f}mm"
        cv2.putText(visualization, text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add position text
        pos_text = measurement['position']
        cv2.putText(visualization, pos_text, (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw lines connecting tooth pairs
    for pair in tooth_pairs:
        primary_pos = pair['primary_molar']['position']
        premolar_pos = pair['premolar']['position']
        
        # Draw line between paired teeth
        cv2.line(visualization, tuple(primary_pos), tuple(premolar_pos), (255, 0, 255), 2)
        
        # Add difference text at midpoint
        mid_x = (primary_pos[0] + premolar_pos[0]) // 2
        mid_y = (primary_pos[1] + premolar_pos[1]) // 2
        diff_text = f"Δ: {pair['width_difference_mm']:.1f}mm"
        cv2.putText(visualization, diff_text, (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return visualization

def process_single_image(image_path, output_dir=None, calibration_factor=0.15, debug=False):
    """
    Process a single dental radiograph image.
    Uses the same working approach as main.py
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
        # Get base filename for outputs
        base_name = Path(image_path).stem
        if debug:
            print(f"  Processing: {base_name}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Failed to load image: {image_path}")
            return None
        
        if debug:
            print(f"  ✅ Image loaded: {image.shape}")
        
        # Create segmentation model
        try:
            segmentation_model = create_segmentation_model()
            if debug:
                print("  ✅ Segmentation model created")
        except Exception as e:
            print(f"  ❌ Failed to create model: {e}")
            return None
        
        # Run segmentation
        try:
            segmentation = segmentation_model.predict(image)
            if debug:
                print(f"  ✅ Segmentation complete: {segmentation.shape}")
        except Exception as e:
            print(f"  ❌ Segmentation failed: {e}")
            return None
        
        # Extract measurements
        try:
            measurements = extract_tooth_measurements(segmentation, calibration_factor=calibration_factor)
            if debug:
                print(f"  ✅ Found {len(measurements)} tooth regions")
        except Exception as e:
            print(f"  ❌ Measurement extraction failed: {e}")
            return None
        
        # Create tooth pairs
        tooth_pairs = create_tooth_pairs(measurements)
        if debug:
            print(f"  ✅ Created {len(tooth_pairs)} tooth pairs")
        
        # Prepare results
        results_data = {
            "image": image_path,
            "processed_date": datetime.now().isoformat(),
            "calibration_factor": calibration_factor,
            "tooth_pairs": tooth_pairs,
            "summary": {
                "total_pairs": len(tooth_pairs),
                "average_difference": np.mean([pair["width_difference_mm"] for pair in tooth_pairs]) if tooth_pairs else 0
            },
            "analysis": {
                "average_ratio": np.mean([pair["primary_molar"]["width_mm"] / pair["premolar"]["width_mm"] 
                                        for pair in tooth_pairs]) if tooth_pairs else 0.0,
                "std_deviation": np.std([pair["width_difference_mm"] for pair in tooth_pairs]) if tooth_pairs else 0.0,
                "valid_pairs": len(tooth_pairs)
            }
        }
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON measurements
            json_path = os.path.join(output_dir, f"{base_name}_measurements.json")
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Create and save visualization
            try:
                visualization = visualize_measurements(image, measurements, tooth_pairs)
                vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
                cv2.imwrite(vis_path, visualization)
                if debug:
                    print(f"  ✅ Saved: {base_name}_measurements.json and {base_name}_visualization.jpg")
            except Exception as e:
                print(f"  ⚠️  Visualization failed: {e}")
        
        # Debug information
        if debug:
            if not tooth_pairs:
                print(f"  ⚠️  No tooth pairs found, but {len(measurements)} individual teeth detected")
                for measurement in measurements[:3]:  # Show first 3
                    print(f"    - {measurement['position']}: {measurement['mesiodistal_width_mm']:.2f}mm")
            else:
                print(f"  📊 Results: {len(tooth_pairs)} pairs, avg diff: {results_data['summary']['average_difference']:.2f}mm")
        
        return results_data
    
    except Exception as e:
        print(f"  ❌ Error processing {image_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return None

def batch_process_directory(input_dir, output_dir=None, calibration_factor=0.15, debug=False):
    """
    Process all dental radiograph images in a directory.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return None
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
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
    successful_results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path.name}")
        
        result = process_single_image(
            str(image_path),
            output_dir,
            calibration_factor,
            debug
        )
        
        all_results.append(result)
        if result and result.get('tooth_pairs'):
            successful_results.append(result)
    
    # Generate summary CSV
    if output_dir:
        csv_path = os.path.join(output_dir, "measurements_summary.csv")
        
        print(f"\n📊 Generating summary CSV...")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                "Image", "Quadrant", "Primary_Molar_Width_mm", 
                "Premolar_Width_mm", "Width_Difference_mm", "Primary_Position", "Premolar_Position"
            ])
            
            # Write data
            total_pairs = 0
            for result in all_results:
                if result and result.get("tooth_pairs"):
                    image_name = Path(result["image"]).name
                    
                    for pair in result["tooth_pairs"]:
                        try:
                            writer.writerow([
                                image_name,
                                pair["quadrant"],
                                f"{pair['primary_molar']['width_mm']:.2f}",
                                f"{pair['premolar']['width_mm']:.2f}",
                                f"{pair['width_difference_mm']:.2f}",
                                f"{pair['primary_molar']['position']}",
                                f"{pair['premolar']['position']}"
                            ])
                            total_pairs += 1
                        except KeyError as e:
                            print(f"Warning: Could not extract data for {image_name}: {e}")
                            continue
        
        print(f"✅ Summary saved to {csv_path}")
        print(f"   Total images processed: {len(all_results)}")
        print(f"   Images with measurements: {len(successful_results)}")
        print(f"   Total tooth pairs found: {total_pairs}")
        
        # Generate simple statistics
        if successful_results:
            all_differences = []
            for result in successful_results:
                for pair in result.get("tooth_pairs", []):
                    all_differences.append(pair["width_difference_mm"])
            
            if all_differences:
                avg_diff = np.mean(all_differences)
                std_diff = np.std(all_differences)
                min_diff = np.min(all_differences)
                max_diff = np.max(all_differences)
                
                print(f"\n📈 Summary Statistics:")
                print(f"   Average width difference: {avg_diff:.2f} ± {std_diff:.2f} mm")
                print(f"   Range: {min_diff:.2f} to {max_diff:.2f} mm")
                print(f"   Success rate: {len(successful_results)}/{len(all_results)} ({100*len(successful_results)/len(all_results):.1f}%)")
    
    print("\nProcessing complete!")
    return all_results

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor Batch Processing')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default='results',
                        help='Path to save the output visualizations and measurements')
    parser.add_argument('--calibration', type=float, default=0.15,
                        help='Calibration factor (mm/pixel), default=0.15')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed progress information')
    
    args = parser.parse_args()
    
    print(f"🦷 Dental Width Predictor - Batch Processing")
    print(f"=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Calibration: {args.calibration} mm/pixel")
    print(f"Debug: {args.debug}")
    print()
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.input):
        results = batch_process_directory(
            args.input,
            args.output,
            args.calibration,
            args.debug
        )
    else:
        results = process_single_image(
            args.input,
            args.output,
            args.calibration,
            args.debug
        )
    
    if not results:
        print("\n❌ No valid results were produced.")
        print("   Try enabling --debug to see detailed error information.")
    else:
        print("\n✅ Batch processing completed successfully!")

if __name__ == "__main__":
    main()
