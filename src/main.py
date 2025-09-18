#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main module for Dental Width Predictor.
Simplified version that works with our new AI models.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our working models
from models.model import create_segmentation_model, extract_tooth_measurements, preprocess_dental_xray

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the dental radiograph image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization')
    parser.add_argument('--calibration', type=float, default=0.15,
                        help='Calibration factor (mm/pixel), default=0.15')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    return parser.parse_args()

def create_tooth_pairs(measurements):
    """
    Group individual tooth measurements into primary molar - premolar pairs
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

def main():
    """Main function to process dental radiographs and measure tooth widths."""
    args = parse_arguments()
    
    # Check if image exists
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Create segmentation model
    try:
        segmentation_model = create_segmentation_model()
        print("✅ Segmentation model created")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return
    
    # Run segmentation
    try:
        segmentation = segmentation_model.predict(image)
        print(f"✅ Segmentation complete: {segmentation.shape}")
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return
    
    # Extract measurements
    try:
        measurements = extract_tooth_measurements(segmentation, calibration_factor=args.calibration)
        print(f"✅ Found {len(measurements)} tooth regions")
    except Exception as e:
        print(f"❌ Measurement extraction failed: {e}")
        return
    
    # Create tooth pairs
    tooth_pairs = create_tooth_pairs(measurements)
    print(f"✅ Created {len(tooth_pairs)} tooth pairs")
    
    # Prepare results
    results_data = {
        "image": image_path,
        "processed_date": datetime.now().isoformat(),
        "calibration_factor": args.calibration,
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
    
    # Print results
    print(f"\n📊 Results:")
    print("=" * 50)
    
    if not tooth_pairs:
        print("No valid tooth pairs found for measurement.")
        print("Individual tooth measurements:")
        for i, measurement in enumerate(measurements):
            print(f"  Tooth {i+1}: {measurement['position']}, "
                  f"width={measurement['mesiodistal_width_mm']:.2f}mm")
    else:
        for i, pair in enumerate(tooth_pairs):
            print(f"Tooth Pair {i+1} ({pair['quadrant']}):")
            print(f"  Primary Molar: {pair['primary_molar']['width_mm']:.2f}mm")
            print(f"  Premolar: {pair['premolar']['width_mm']:.2f}mm")
            print(f"  Difference: {pair['width_difference_mm']:.2f}mm")
            print()
        
        print(f"Summary:")
        print(f"  Total pairs: {results_data['summary']['total_pairs']}")
        print(f"  Average difference: {results_data['summary']['average_difference']:.2f}mm")
    
    # Create visualization
    try:
        visualization = visualize_measurements(image, measurements, tooth_pairs)
        print("✅ Visualization created")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        visualization = image.copy()
    
    # Save results
    if args.output:
        # Save visualization
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        cv2.imwrite(args.output, visualization)
        print(f"✅ Visualization saved: {args.output}")
        
        # Save JSON measurements
        base_name = os.path.splitext(args.output)[0]
        json_path = f"{base_name}_measurements.json"
        
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"✅ Measurements saved: {json_path}")
    
    # Debug information
    if args.debug:
        print(f"\n🔍 Debug Information:")
        print(f"  Total measurements: {len(measurements)}")
        print(f"  Calibration factor: {args.calibration} mm/pixel")
        print(f"  Image dimensions: {image.shape}")
        for measurement in measurements:
            print(f"  - {measurement['position']}: "
                  f"{measurement['mesiodistal_width_mm']:.2f}mm "
                  f"(area: {measurement['area']:.0f})")

if __name__ == "__main__":
    main()
