#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Focused E-Space Detection System

This system ONLY detects and measures 2nd primary molars and 2nd premolars
in the posterior regions for E-space quantification, filtering out all anterior teeth.
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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_processing import preprocess_image, enhance_teeth_boundaries
from src.detection.tooth_detection import detect_teeth, merge_overlapping_contours
from src.measurement.width_measurement import measure_contact_points


@dataclass
class ESpaceResult:
    """E-space measurement result for a quadrant."""
    quadrant: str  # 'upper_left', 'upper_right', 'lower_left', 'lower_right'
    primary_molar_width: float  # mm
    premolar_width: float  # mm
    e_space: float  # mm (primary - premolar)
    primary_molar_bbox: Tuple[int, int, int, int]  # x, y, w, h
    premolar_bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float  # 0-1


class FocusedESpaceDetector:
    """Detector focused specifically on E-space relevant teeth."""
    
    def __init__(self, calibration_factor=0.1):
        self.calibration_factor = calibration_factor
        
    def detect_posterior_teeth_only(self, image, debug_dir=None):
        """
        Detect ONLY teeth in posterior regions, filtering out all anterior teeth.
        
        Returns:
            list: Contours of teeth in posterior regions only
        """
        height, width = image.shape[:2]
        
        # Create posterior region masks (MUCH MORE RESTRICTIVE)
        posterior_mask = self._create_strict_posterior_mask(image)
        
        # Apply multiple detection strategies with posterior filtering
        all_contours = []
        
        # Strategy 1: Standard detection with posterior filtering
        contours1 = self._detect_with_posterior_filter(image, posterior_mask, "standard")
        
        # Strategy 2: Enhanced for small erupting teeth
        contours2 = self._detect_with_posterior_filter(image, posterior_mask, "small_teeth")
        
        # Strategy 3: Enhanced for large primary molars
        contours3 = self._detect_with_posterior_filter(image, posterior_mask, "large_teeth")
        
        # Combine and deduplicate
        all_contours = contours1 + contours2 + contours3
        if all_contours:
            all_contours = merge_overlapping_contours(all_contours, threshold=0.3)
        
        # STRICT filtering - only keep teeth in posterior regions
        filtered_contours = self._strict_posterior_filter(image, all_contours)
        
        # Save debug visualization
        if debug_dir:
            self._save_posterior_detection_debug(image, filtered_contours, posterior_mask, debug_dir)
        
        return filtered_contours
    
    def _create_strict_posterior_mask(self, image):
        """Create a very restrictive mask for only posterior teeth regions."""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define very specific posterior regions
        center_x = width // 2
        
        # Upper and lower jaw levels
        upper_y = int(height * 0.35)  # Upper jaw posterior region
        lower_y = int(height * 0.65)  # Lower jaw posterior region
        
        # Posterior region dimensions (smaller and more focused)
        region_width = int(width * 0.15)   # 15% of image width for each posterior region
        region_height = int(height * 0.12)  # 12% of image height
        
        # Define 4 posterior regions (one for each quadrant)
        regions = [
            # Upper left posterior (2nd premolar/2nd primary molar area)
            (int(center_x * 0.25), upper_y, region_width, region_height),
            # Upper right posterior
            (int(center_x * 1.75) - region_width, upper_y, region_width, region_height),
            # Lower left posterior  
            (int(center_x * 0.25), lower_y, region_width, region_height),
            # Lower right posterior
            (int(center_x * 1.75) - region_width, lower_y, region_width, region_height)
        ]
        
        # Draw rectangular regions
        for x, y, w, h in regions:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        return mask
    
    def _detect_with_posterior_filter(self, image, posterior_mask, strategy):
        """Detect teeth with specific strategy and posterior filtering."""
        if strategy == "standard":
            processed = image
            sigma1, sigma2 = 1.5, 2.5
            area_range = (400, 8000)
        elif strategy == "small_teeth":
            # Enhanced for small erupting premolars
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
            processed = clahe.apply(image)
            sigma1, sigma2 = 1.0, 3.5
            area_range = (200, 4000)
        else:  # large_teeth
            # Smoothed for large primary molars
            processed = cv2.GaussianBlur(image, (3, 3), 0)
            sigma1, sigma2 = 2.0, 3.0
            area_range = (800, 12000)
        
        # Edge detection
        from skimage import feature, exposure, morphology, measure
        
        p2, p98 = np.percentile(processed, (2, 98))
        enhanced = exposure.rescale_intensity(processed, in_range=(p2, p98))
        
        edges1 = feature.canny(enhanced, sigma=sigma1)
        edges2 = feature.canny(enhanced, sigma=sigma2)
        edges = np.logical_or(edges1, edges2)
        
        # Apply posterior mask IMMEDIATELY
        edges = edges & (posterior_mask > 0)
        
        # Morphological operations
        dilated = morphology.dilation(edges, morphology.disk(2))
        filled = morphology.remove_small_holes(dilated, area_threshold=50)
        closed = morphology.closing(filled, morphology.disk(3))
        
        # Find contours
        labeled = measure.label(closed)
        props = measure.regionprops(labeled)
        
        valid_contours = []
        for prop in props:
            if (area_range[0] <= prop.area <= area_range[1] and
                prop.eccentricity >= 0.15 and  # Not too round
                prop.solidity >= 0.45 and      # Not too irregular
                self._is_posterior_tooth_shape(prop)):
                
                binary = (labeled == prop.label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) >= area_range[0]:
                        valid_contours.append(cnt)
        
        return valid_contours
    
    def _strict_posterior_filter(self, image, contours):
        """Apply very strict filtering to keep only posterior teeth."""
        height, width = image.shape[:2]
        center_x = width // 2
        
        filtered = []
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Very strict position filtering
            is_posterior = False
            
            # Left posterior (must be significantly left of center)
            if cx < center_x * 0.6 and cx > center_x * 0.1:
                # Must be in upper or lower jaw regions
                if (height * 0.25 < cy < height * 0.45) or (height * 0.55 < cy < height * 0.75):
                    is_posterior = True
            
            # Right posterior (must be significantly right of center)  
            elif cx > center_x * 1.4 and cx < center_x * 1.9:
                if (height * 0.25 < cy < height * 0.45) or (height * 0.55 < cy < height * 0.75):
                    is_posterior = True
            
            if is_posterior:
                filtered.append(contour)
        
        return filtered
    
    def _is_posterior_tooth_shape(self, prop):
        """Check if shape is consistent with posterior tooth."""
        # Posterior teeth characteristics
        area_ratio = prop.area / prop.bbox_area if prop.bbox_area > 0 else 0
        aspect_ratio = prop.minor_axis_length / prop.major_axis_length if prop.major_axis_length > 0 else 0
        
        # More selective shape criteria
        return (0.4 < area_ratio < 0.85 and  # Not too sparse, not too filled
                0.4 < aspect_ratio < 0.9 and  # Not too elongated, not perfectly round
                prop.extent > 0.5)             # Good fill of bounding box
    
    def classify_posterior_teeth(self, image, contours, debug_dir=None):
        """
        Classify posterior teeth as 2nd primary molars or 2nd premolars.
        """
        classified_teeth = []
        height, width = image.shape[:2]
        center_x = width // 2
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Determine quadrant
            if cx < center_x and cy < height // 2:
                quadrant = "upper_left"
            elif cx >= center_x and cy < height // 2:
                quadrant = "upper_right"
            elif cx < center_x:
                quadrant = "lower_left"
            else:
                quadrant = "lower_right"
            
            # Classify as primary molar vs premolar
            tooth_type = self._classify_primary_vs_premolar(image, contour, area)
            
            tooth_data = {
                'contour': contour,
                'type': tooth_type,
                'quadrant': quadrant,
                'centroid': (cx, cy),
                'area': area,
                'bbox': (x, y, w, h)
            }
            
            classified_teeth.append(tooth_data)
        
        if debug_dir:
            self._save_classification_debug(image, classified_teeth, debug_dir)
        
        return classified_teeth
    
    def _classify_primary_vs_premolar(self, image, contour, area):
        """
        Classify as 2nd primary molar or 2nd premolar based on size and characteristics.
        
        Key differences:
        - 2nd primary molars: larger, more calcified (brighter), more established
        - 2nd premolars: smaller, may be less calcified, more oval shape
        """
        # Extract region for analysis
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        roi = cv2.bitwise_and(image, image, mask=mask)
        roi_pixels = roi[mask > 0]
        
        if len(roi_pixels) == 0:
            return "unknown"
        
        # Feature calculation
        mean_intensity = np.mean(roi_pixels)
        
        # Size-based primary classification
        if area > 1500:  # Large - likely primary molar
            return "second_primary_molar"
        elif area < 800:  # Small - likely premolar
            return "second_premolar"
        else:  # Medium size - use intensity
            if mean_intensity > 100:  # More calcified
                return "second_primary_molar"
            else:
                return "second_premolar"
    
    def calculate_e_space(self, image, classified_teeth):
        """
        Calculate E-space for each quadrant where we have both tooth types.
        """
        e_space_results = []
        
        # Group by quadrant
        quadrants = {}
        for tooth in classified_teeth:
            quadrant = tooth['quadrant']
            if quadrant not in quadrants:
                quadrants[quadrant] = []
            quadrants[quadrant].append(tooth)
        
        # Calculate E-space for each quadrant
        for quadrant, teeth in quadrants.items():
            primary_molars = [t for t in teeth if t['type'] == 'second_primary_molar']
            premolars = [t for t in teeth if t['type'] == 'second_premolar']
            
            # Find the best pair in this quadrant
            best_pair = self._find_best_tooth_pair(primary_molars, premolars)
            
            if best_pair:
                primary_molar, premolar = best_pair
                
                # Measure both teeth
                primary_measurement = measure_contact_points(image, type('obj', (object,), {
                    'contour': primary_molar['contour'],
                    'bounding_box': primary_molar['bbox'],
                    'centroid': primary_molar['centroid']
                })(), self.calibration_factor)
                
                premolar_measurement = measure_contact_points(image, type('obj', (object,), {
                    'contour': premolar['contour'],
                    'bounding_box': premolar['bbox'],
                    'centroid': premolar['centroid']
                })(), self.calibration_factor)
                
                if primary_measurement and premolar_measurement:
                    primary_width = primary_measurement['width']
                    premolar_width = premolar_measurement['width']
                    e_space = primary_width - premolar_width
                    
                    result = ESpaceResult(
                        quadrant=quadrant,
                        primary_molar_width=primary_width,
                        premolar_width=premolar_width,
                        e_space=e_space,
                        primary_molar_bbox=primary_molar['bbox'],
                        premolar_bbox=premolar['bbox'],
                        confidence=0.8  # Can be improved with more sophisticated metrics
                    )
                    
                    e_space_results.append(result)
        
        return e_space_results
    
    def _find_best_tooth_pair(self, primary_molars, premolars):
        """Find the best primary molar - premolar pair in a quadrant."""
        if not primary_molars or not premolars:
            return None
        
        best_pair = None
        min_distance = float('inf')
        
        for primary in primary_molars:
            for premolar in premolars:
                # Calculate distance
                p_cx, p_cy = primary['centroid']
                pm_cx, pm_cy = premolar['centroid']
                
                distance = np.sqrt((p_cx - pm_cx)**2 + (p_cy - pm_cy)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (primary, premolar)
        
        # Only return if teeth are reasonably close
        if min_distance < 150:  # pixels
            return best_pair
        
        return None
    
    def _save_posterior_detection_debug(self, image, contours, posterior_mask, debug_dir):
        """Save debug image showing posterior detection."""
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw posterior mask in blue
        mask_colored = cv2.cvtColor(posterior_mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:,:,0] = posterior_mask  # Blue channel
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_colored, 0.3, 0)
        
        # Draw detected contours in green
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)
        
        # Add text overlay
        cv2.putText(debug_img, f"Posterior teeth detected: {len(contours)}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, "Blue: Posterior regions, Green: Detected teeth", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        debug_path = os.path.join(debug_dir, "posterior_detection.jpg")
        cv2.imwrite(debug_path, debug_img)
    
    def _save_classification_debug(self, image, classified_teeth, debug_dir):
        """Save debug image showing tooth classification."""
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        colors = {
            'second_primary_molar': (0, 255, 0),   # Green
            'second_premolar': (255, 0, 0),        # Blue
            'unknown': (0, 255, 255)               # Yellow
        }
        
        for tooth in classified_teeth:
            color = colors.get(tooth['type'], (128, 128, 128))
            cv2.drawContours(debug_img, [tooth['contour']], -1, color, 3)
            
            # Add label
            cx, cy = tooth['centroid']
            label = "2nd PM" if tooth['type'] == 'second_primary_molar' else "2nd Pre"
            cv2.putText(debug_img, label, (cx-20, cy-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        debug_path = os.path.join(debug_dir, "posterior_classification.jpg")
        cv2.imwrite(debug_path, debug_img)
    
    def visualize_e_space_results(self, image, e_space_results):
        """Create visualization of E-space measurements."""
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for result in e_space_results:
            # Draw bounding boxes
            x1, y1, w1, h1 = result.primary_molar_bbox
            x2, y2, w2, h2 = result.premolar_bbox
            
            # Primary molar in green
            cv2.rectangle(vis_img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 3)
            cv2.putText(vis_img, f"2nd PM: {result.primary_molar_width:.1f}mm", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Premolar in blue
            cv2.rectangle(vis_img, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 3)
            cv2.putText(vis_img, f"2nd Pre: {result.premolar_width:.1f}mm", 
                       (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # E-space result
            mid_x = (x1 + x2 + w1 + w2) // 4
            mid_y = (y1 + y2 + h1 + h2) // 4
            cv2.putText(vis_img, f"E-space: {result.e_space:.1f}mm", 
                       (mid_x-50, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Draw connection line
            center1 = (x1 + w1//2, y1 + h1//2)
            center2 = (x2 + w2//2, y2 + h2//2)
            cv2.line(vis_img, center1, center2, (255, 255, 0), 2)
        
        return vis_img


def process_e_space_image(image_path, output_dir=None, calibration_factor=0.1, debug=False):
    """
    Process a single image for E-space analysis.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    
    # Load and preprocess
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Failed to load image: {image_path}"}
    
    processed_image = preprocess_image(image)
    
    # Create output directory
    base_name = Path(image_path).stem
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    debug_dir = None
    if debug and output_dir:
        debug_dir = os.path.join(output_dir, f"{base_name}_e_space_debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize E-space detector
    detector = FocusedESpaceDetector(calibration_factor)
    
    # Step 1: Detect only posterior teeth
    posterior_contours = detector.detect_posterior_teeth_only(processed_image, debug_dir)
    
    if not posterior_contours:
        return {
            "image": image_path,
            "success": False,
            "error": "No posterior teeth detected",
            "e_space_measurements": []
        }
    
    # Step 2: Classify as 2nd primary molars vs 2nd premolars
    classified_teeth = detector.classify_posterior_teeth(processed_image, posterior_contours, debug_dir)
    
    # Step 3: Calculate E-space
    e_space_results = detector.calculate_e_space(processed_image, classified_teeth)
    
    # Create visualization
    if e_space_results:
        vis_image = detector.visualize_e_space_results(image, e_space_results)
        if output_dir:
            vis_path = os.path.join(output_dir, f"{base_name}_e_space_results.jpg")
            cv2.imwrite(vis_path, vis_image)
    
    # Prepare results
    result = {
        "image": image_path,
        "processed_date": datetime.now().isoformat(),
        "success": len(e_space_results) > 0,
        "posterior_teeth_detected": len(posterior_contours),
        "classified_teeth": len(classified_teeth),
        "e_space_measurements": []
    }
    
    # Add E-space measurements
    for e_result in e_space_results:
        result["e_space_measurements"].append({
            "quadrant": e_result.quadrant,
            "second_primary_molar_width_mm": round(e_result.primary_molar_width, 2),
            "second_premolar_width_mm": round(e_result.premolar_width, 2),
            "e_space_mm": round(e_result.e_space, 2),
            "confidence": round(e_result.confidence, 2)
        })
    
    # Save JSON results
    if output_dir:
        json_path = os.path.join(output_dir, f"{base_name}_e_space.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    return result


def main():
    """Main function for E-space analysis."""
    parser = argparse.ArgumentParser(description='E-Space Quantification for Dental Radiographs')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default='e_space_results',
                        help='Path to save the results')
    parser.add_argument('--calibration', type=float, default=0.1,
                        help='Calibration factor (mm/pixel), default=0.1')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with visualization')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single image
        result = process_e_space_image(args.input, args.output, args.calibration, args.debug)
        
        if result.get("success"):
            print(f"✅ E-space analysis complete!")
            for measurement in result["e_space_measurements"]:
                print(f"  {measurement['quadrant']}: E-space = {measurement['e_space_mm']:.1f}mm")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    else:
        # Batch processing
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(args.input).glob(f"*{ext.lower()}")))
            image_paths.extend(list(Path(args.input).glob(f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"No images found in {args.input}")
            return
        
        print(f"Processing {len(image_paths)} images for E-space analysis...")
        
        # Process all images
        results = []
        successful = 0
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            result = process_e_space_image(str(image_path), args.output, args.calibration, args.debug)
            results.append(result)
            
            if result.get("success"):
                successful += 1
                measurements = result["e_space_measurements"]
                print(f"  ✅ Found {len(measurements)} E-space measurements")
                for m in measurements:
                    print(f"    {m['quadrant']}: {m['e_space_mm']:.1f}mm")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Create summary CSV
        if args.output:
            csv_path = os.path.join(args.output, "e_space_summary.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Image", "Quadrant", "2nd Primary Molar (mm)", 
                    "2nd Premolar (mm)", "E-Space (mm)", "Success"
                ])
                
                for result in results:
                    image_name = Path(result["image"]).name
                    if result.get("success") and result["e_space_measurements"]:
                        for m in result["e_space_measurements"]:
                            writer.writerow([
                                image_name, m["quadrant"],
                                m["second_primary_molar_width_mm"],
                                m["second_premolar_width_mm"],
                                m["e_space_mm"], "Yes"
                            ])
                    else:
                        writer.writerow([image_name, "", "", "", "", "No"])
            
            print(f"\n📊 Summary saved to: {csv_path}")
        
        print(f"\n🏁 Batch processing complete!")
        print(f"   Total images: {len(image_paths)}")
        print(f"   Successful: {successful}")
        print(f"   Success rate: {successful/len(image_paths)*100:.1f}%")


if __name__ == "__main__":
    main()
