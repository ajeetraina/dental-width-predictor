#!/usr/bin/env python3
"""
Debug script to understand why tooth detection is not working
"""

import cv2
import numpy as np
import os
import json
from models.model import create_segmentation_model, extract_tooth_measurements, preprocess_dental_xray

def debug_image_processing(image_path):
    """Debug the complete image processing pipeline"""
    
    print(f"\n🔍 Debugging: {os.path.basename(image_path)}")
    print("="*60)
    
    # Step 1: Load original image
    original = cv2.imread(image_path)
    if original is None:
        print("❌ Failed to load image!")
        return False
    
    print(f"✅ Original image loaded: {original.shape}")
    
    # Step 2: Preprocessing  
    try:
        processed = preprocess_dental_xray(image_path)
        print(f"✅ Preprocessing successful: {processed.shape}")
        print(f"   Pixel range: {processed.min():.3f} - {processed.max():.3f}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False
    
    # Step 3: Convert to grayscale for OpenCV processing
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original
    
    print(f"✅ Grayscale conversion: {gray.shape}")
    
    # Step 4: Enhanced preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    print(f"✅ CLAHE enhancement applied")
    
    # Step 5: Edge detection with different parameters
    edges_params = [
        (50, 150, "Standard"),
        (30, 100, "Sensitive"), 
        (70, 200, "Conservative"),
        (20, 80, "Very Sensitive")
    ]
    
    best_contour_count = 0
    best_edges = None
    best_params = None
    
    for low, high, name in edges_params:
        edges = cv2.Canny(enhanced, low, high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum tooth area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Tooth-like aspect ratio (more permissive)
                    valid_contours.append(contour)
        
        print(f"   {name} ({low}-{high}): {len(contours)} total, {len(valid_contours)} valid")
        
        if len(valid_contours) > best_contour_count:
            best_contour_count = len(valid_contours)
            best_edges = edges
            best_params = (low, high, name)
    
    print(f"🏆 Best parameters: {best_params[2]} with {best_contour_count} valid contours")
    
    # Step 6: Create segmentation model and test
    try:
        model = create_segmentation_model()
        print("✅ Segmentation model created")
        
        # Test prediction
        segmentation = model.predict(original)
        print(f"✅ Segmentation prediction: {segmentation.shape}")
        print(f"   Segmentation range: {segmentation.min():.3f} - {segmentation.max():.3f}")
        
        # Check if segmentation has any tooth regions
        tooth_pixels = np.sum(segmentation[:,:,1:])  # Sum of tooth channels
        print(f"   Total tooth pixels detected: {tooth_pixels:.0f}")
        
        # Extract measurements
        measurements = extract_tooth_measurements(segmentation, calibration_factor=0.1)
        print(f"✅ Measurements extracted: {len(measurements)} regions found")
        
        for i, measurement in enumerate(measurements):
            print(f"   Region {i+1}: {measurement['position']}, "
                  f"width={measurement['mesiodistal_width_mm']:.2f}mm, "
                  f"area={measurement['area']:.0f}")
            
    except Exception as e:
        print(f"❌ Model processing failed: {e}")
        return False
    
    # Step 7: Try more permissive detection
    print(f"\n🔧 Trying more permissive detection...")
    
    # Use the best edge detection
    if best_edges is not None:
        # Morphological operations to fill gaps
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(best_edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Very permissive filtering
        permissive_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Lower area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:  # Very permissive aspect ratio
                    permissive_contours.append(contour)
                    width_mm = w * 0.1  # Default calibration
                    print(f"   Permissive detection: area={area:.0f}, "
                          f"size={w}x{h}, width={width_mm:.2f}mm")
        
        print(f"✅ Permissive method found: {len(permissive_contours)} regions")
    
    return len(measurements) > 0 or best_contour_count > 0

def test_multiple_images():
    """Test multiple sample images to understand the pattern"""
    
    sample_dir = "/app/data/samples"
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n🔍 Testing {len(sample_files)} sample images...")
    print("="*60)
    
    successful_detections = 0
    
    for i, filename in enumerate(sample_files[:5]):  # Test first 5 images
        image_path = os.path.join(sample_dir, filename)
        print(f"\n📸 Image {i+1}/5: {filename}")
        
        if debug_image_processing(image_path):
            successful_detections += 1
    
    print(f"\n📊 Summary: {successful_detections}/5 images had detections")
    
    if successful_detections == 0:
        print("\n💡 Recommendations:")
        print("1. Dental X-rays may need specialized preprocessing")
        print("2. Consider adjusting calibration factor (currently 0.1 mm/pixel)")
        print("3. May need to train on actual dental data")
        print("4. Try different edge detection thresholds")

if __name__ == "__main__":
    print("🦷 Dental Width Predictor - Debug Mode")
    print("="*60)
    
    # Test 1: Model imports
    try:
        from models import create_segmentation_model, create_tooth_classifier
        print("✅ Models imported successfully")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        exit(1)
    
    # Test 2: Sample images availability  
    sample_dir = "/app/data/samples"
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory not found: {sample_dir}")
        exit(1)
    
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Found {len(sample_files)} sample images")
    
    if not sample_files:
        print("❌ No sample images found!")
        exit(1)
    
    # Test 3: Individual image processing
    test_multiple_images()
    
    print(f"\n🎯 Debug complete!")
