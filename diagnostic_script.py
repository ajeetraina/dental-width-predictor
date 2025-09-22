#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic script for Dental Width Predictor.

This script helps identify issues with the current setup and provides
recommendations for fixing batch processing problems.
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 50)
    print("CHECKING DEPENDENCIES")
    print("=" * 50)
    
    dependencies = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'scikit-image': 'skimage',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    missing_deps = []
    
    for package, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package}: OK")
        except ImportError:
            print(f"✗ {package}: MISSING")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n⚠ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("\n✅ All dependencies are installed")
    return True


def check_ml_models():
    """Check if ML models are available."""
    print("\n" + "=" * 50)
    print("CHECKING ML MODELS")
    print("=" * 50)
    
    model_files = [
        "models/model.py",
        "models/segmentation_model.h5",
        "models/tooth_classifier.h5",
        "src/detection/ml_tooth_detection.py",
        "src/detection/ml_tooth_classification.py"
    ]
    
    available_models = []
    missing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ {model_file}: EXISTS")
            available_models.append(model_file)
        else:
            print(f"✗ {model_file}: MISSING")
            missing_models.append(model_file)
    
    # Try to import ML modules
    try:
        from src.detection.ml_tooth_detection import detect_teeth_ml
        print("✓ ML tooth detection: IMPORTABLE")
    except ImportError as e:
        print(f"✗ ML tooth detection: NOT IMPORTABLE ({e})")
    
    try:
        from src.detection.ml_tooth_classification import classify_teeth_ml
        print("✓ ML tooth classification: IMPORTABLE")
    except ImportError as e:
        print(f"✗ ML tooth classification: NOT IMPORTABLE ({e})")
    
    if missing_models:
        print(f"\n⚠ Missing ML models - will use traditional methods only")
        return False
    else:
        print(f"\n✅ ML models are available")
        return True


def analyze_sample_images(input_dir, max_samples=5):
    """Analyze sample images to understand data characteristics."""
    print("\n" + "=" * 50)
    print("ANALYZING SAMPLE IMAGES")
    print("=" * 50)
    
    if not os.path.exists(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        return {}
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.lower()}")))
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"✗ No images found in {input_dir}")
        return {}
    
    print(f"Found {len(image_paths)} images")
    
    # Analyze sample images
    analysis_results = {
        "total_images": len(image_paths),
        "analyzed_samples": min(max_samples, len(image_paths)),
        "image_stats": [],
        "common_issues": []
    }
    
    sample_paths = image_paths[:max_samples]
    
    for i, image_path in enumerate(sample_paths):
        print(f"\nAnalyzing sample {i+1}: {Path(image_path).name}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  ✗ Failed to load image")
                analysis_results["common_issues"].append(f"Failed to load: {Path(image_path).name}")
                continue
            
            # Basic image statistics
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if channels == 3 else image
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            min_intensity = np.min(gray)
            max_intensity = np.max(gray)
            
            # Check for common issues
            issues = []
            
            # Too dark or too bright
            if mean_intensity < 50:
                issues.append("Very dark image")
            elif mean_intensity > 200:
                issues.append("Very bright image")
            
            # Low contrast
            if std_intensity < 30:
                issues.append("Low contrast")
            
            # Check resolution
            if width < 500 or height < 500:
                issues.append("Low resolution")
            elif width > 5000 or height > 5000:
                issues.append("Very high resolution")
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                issues.append("Unusual aspect ratio")
            
            sample_stats = {
                "filename": Path(image_path).name,
                "dimensions": (width, height),
                "channels": channels,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "intensity_range": (min_intensity, max_intensity),
                "issues": issues
            }
            
            analysis_results["image_stats"].append(sample_stats)
            
            print(f"  Dimensions: {width}x{height}")
            print(f"  Intensity: μ={mean_intensity:.1f}, σ={std_intensity:.1f}")
            print(f"  Range: [{min_intensity}, {max_intensity}]")
            
            if issues:
                print(f"  ⚠ Issues: {', '.join(issues)}")
                analysis_results["common_issues"].extend(issues)
            else:
                print(f"  ✓ No obvious issues detected")
                
        except Exception as e:
            print(f"  ✗ Error analyzing image: {str(e)}")
            analysis_results["common_issues"].append(f"Analysis error: {Path(image_path).name}")
    
    return analysis_results


def test_detection_pipeline(input_dir, max_tests=3):
    """Test the detection pipeline on sample images."""
    print("\n" + "=" * 50)
    print("TESTING DETECTION PIPELINE")
    print("=" * 50)
    
    # Import required modules
    try:
        from src.preprocessing.image_processing import preprocess_image
        from src.detection.tooth_detection import detect_teeth
        from src.detection.tooth_classification import classify_teeth
        from src.measurement.width_measurement import measure_tooth_width
        print("✓ All modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return {}
    
    # Get sample images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.lower()}")))
        image_paths.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"✗ No images found for testing in {input_dir}")
        return {}
    
    test_results = {
        "tested_images": 0,
        "pipeline_results": [],
        "common_failures": []
    }
    
    sample_paths = image_paths[:max_tests]
    
    for i, image_path in enumerate(sample_paths):
        print(f"\nTesting pipeline on {i+1}: {Path(image_path).name}")
        
        pipeline_result = {
            "filename": Path(image_path).name,
            "steps": {}
        }
        
        try:
            # Step 1: Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  ✗ Failed to load image")
                pipeline_result["steps"]["load"] = {"success": False, "error": "Failed to load"}
                continue
            
            print(f"  ✓ Image loaded: {image.shape}")
            pipeline_result["steps"]["load"] = {"success": True, "shape": image.shape}
            
            # Step 2: Preprocessing
            try:
                processed = preprocess_image(image)
                print(f"  ✓ Preprocessing complete")
                pipeline_result["steps"]["preprocess"] = {"success": True}
            except Exception as e:
                print(f"  ✗ Preprocessing failed: {str(e)}")
                pipeline_result["steps"]["preprocess"] = {"success": False, "error": str(e)}
                continue
            
            # Step 3: Tooth detection
            try:
                contours = detect_teeth(processed)
                print(f"  ✓ Detection complete: {len(contours)} contours found")
                pipeline_result["steps"]["detection"] = {"success": True, "contours": len(contours)}
                
                if len(contours) == 0:
                    print(f"    ⚠ No contours detected - this is a common issue")
                    test_results["common_failures"].append("No contours detected")
                
            except Exception as e:
                print(f"  ✗ Detection failed: {str(e)}")
                pipeline_result["steps"]["detection"] = {"success": False, "error": str(e)}
                continue
            
            # Step 4: Classification
            try:
                if contours:
                    classified = classify_teeth(processed, contours)
                    primary_molars = [t for t in classified if t.type == "primary_molar"]
                    premolars = [t for t in classified if t.type == "premolar"]
                    
                    print(f"  ✓ Classification complete: {len(primary_molars)} primary molars, {len(premolars)} premolars")
                    pipeline_result["steps"]["classification"] = {
                        "success": True, 
                        "primary_molars": len(primary_molars), 
                        "premolars": len(premolars)
                    }
                    
                    if len(primary_molars) == 0 or len(premolars) == 0:
                        print(f"    ⚠ Missing tooth types - cannot form pairs")
                        test_results["common_failures"].append("Missing required tooth types")
                else:
                    print(f"  ⚠ Skipping classification - no contours")
                    pipeline_result["steps"]["classification"] = {"success": False, "error": "No contours"}
                    continue
                    
            except Exception as e:
                print(f"  ✗ Classification failed: {str(e)}")
                pipeline_result["steps"]["classification"] = {"success": False, "error": str(e)}
                continue
            
            # Step 5: Measurement
            try:
                if classified:
                    measurements = measure_tooth_width(processed, classified, 0.1)
                    print(f"  ✓ Measurement complete: {len(measurements)} pairs measured")
                    pipeline_result["steps"]["measurement"] = {"success": True, "pairs": len(measurements)}
                    
                    if len(measurements) == 0:
                        print(f"    ⚠ No measurement pairs found")
                        test_results["common_failures"].append("No measurement pairs formed")
                else:
                    print(f"  ⚠ Skipping measurement - no classified teeth")
                    pipeline_result["steps"]["measurement"] = {"success": False, "error": "No classified teeth"}
                    
            except Exception as e:
                print(f"  ✗ Measurement failed: {str(e)}")
                pipeline_result["steps"]["measurement"] = {"success": False, "error": str(e)}
            
            test_results["pipeline_results"].append(pipeline_result)
            test_results["tested_images"] += 1
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {str(e)}")
            pipeline_result["steps"]["unexpected"] = {"success": False, "error": str(e)}
    
    return test_results


def generate_diagnostic_report(input_dir, output_dir="diagnostic_results"):
    """Generate a comprehensive diagnostic report."""
    print("\n" + "=" * 50)
    print("GENERATING DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run all diagnostics
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": input_dir,
        "dependencies": check_dependencies(),
        "ml_models": check_ml_models(),
        "image_analysis": analyze_sample_images(input_dir),
        "pipeline_test": test_detection_pipeline(input_dir)
    }
    
    # Save detailed report
    report_path = os.path.join(output_dir, "diagnostic_report.json")
    with open(report_path, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    # Generate recommendations
    recommendations = generate_recommendations(diagnostics)
    
    # Save recommendations
    rec_path = os.path.join(output_dir, "recommendations.txt")
    with open(rec_path, 'w') as f:
        f.write("DENTAL WIDTH PREDICTOR - DIAGNOSTIC RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n\n")
        for category, recs in recommendations.items():
            f.write(f"{category.upper().replace('_', ' ')}:\n")
            for rec in recs:
                f.write(f"  • {rec}\n")
            f.write("\n")
    
    print(f"\n✅ Diagnostic report saved to: {report_path}")
    print(f"✅ Recommendations saved to: {rec_path}")
    
    return diagnostics, recommendations


def generate_recommendations(diagnostics):
    """Generate recommendations based on diagnostic results."""
    recommendations = {
        "immediate_actions": [],
        "image_preprocessing": [],
        "detection_improvements": [],
        "calibration_suggestions": [],
        "general_improvements": []
    }
    
    # Check dependencies
    if not diagnostics["dependencies"]:
        recommendations["immediate_actions"].append(
            "Install missing dependencies before proceeding"
        )
    
    # Check ML models
    if not diagnostics["ml_models"]:
        recommendations["immediate_actions"].append(
            "Consider implementing or obtaining ML models for better detection accuracy"
        )
        recommendations["detection_improvements"].append(
            "Current system relies on traditional computer vision methods only"
        )
    
    # Analyze image issues
    image_analysis = diagnostics.get("image_analysis", {})
    common_issues = image_analysis.get("common_issues", [])
    
    if "Very dark image" in common_issues:
        recommendations["image_preprocessing"].append(
            "Apply stronger contrast enhancement for dark images"
        )
    
    if "Very bright image" in common_issues:
        recommendations["image_preprocessing"].append(
            "Implement exposure correction for overexposed images"
        )
    
    if "Low contrast" in common_issues:
        recommendations["image_preprocessing"].append(
            "Increase CLAHE parameters for better contrast enhancement"
        )
    
    if "Low resolution" in common_issues:
        recommendations["image_preprocessing"].append(
            "Consider image upscaling or adjusting detection parameters for low-res images"
        )
    
    # Analyze pipeline failures
    pipeline_test = diagnostics.get("pipeline_test", {})
    common_failures = pipeline_test.get("common_failures", [])
    
    if "No contours detected" in common_failures:
        recommendations["detection_improvements"].extend([
            "Adjust edge detection parameters (sigma values) to be more sensitive",
            "Reduce minimum area thresholds for tooth detection",
            "Implement multi-scale detection approaches",
            "Add adaptive preprocessing based on image characteristics"
        ])
    
    if "Missing required tooth types" in common_failures:
        recommendations["detection_improvements"].extend([
            "Improve tooth classification algorithms",
            "Implement fallback classification strategies",
            "Add manual tooth type assignment options",
            "Review classification criteria and thresholds"
        ])
    
    if "No measurement pairs formed" in common_failures:
        recommendations["detection_improvements"].extend([
            "Implement more flexible tooth pairing algorithms",
            "Add distance-based pairing for isolated teeth",
            "Improve spatial relationship analysis between teeth"
        ])
    
    # Calibration suggestions
    recommendations["calibration_suggestions"].extend([
        "Consider implementing automatic calibration from image metadata",
        "Add manual calibration tools for users",
        "Test different calibration factors for your specific dataset",
        "Implement calibration validation using known measurements"
    ])
    
    # General improvements
    recommendations["general_improvements"].extend([
        "Use the enhanced batch processing script for better error handling",
        "Enable debug mode to inspect intermediate results",
        "Consider manual quality assessment of sample results",
        "Implement validation using ground truth measurements",
        "Add user interface for parameter adjustment"
    ])
    
    return recommendations


def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dental Width Predictor Diagnostic Tool')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to directory containing sample images')
    parser.add_argument('--output', type=str, default='diagnostic_results',
                        help='Path to save diagnostic results')
    
    args = parser.parse_args()
    
    print("DENTAL WIDTH PREDICTOR - DIAGNOSTIC TOOL")
    print("This tool will analyze your setup and identify potential issues")
    print("with batch processing.\n")
    
    # Run comprehensive diagnostics
    diagnostics, recommendations = generate_diagnostic_report(args.input, args.output)
    
    # Print key recommendations
    print("\n" + "=" * 50)
    print("KEY RECOMMENDATIONS")
    print("=" * 50)
    
    for category, recs in recommendations.items():
        if recs:  # Only show categories with recommendations
            print(f"\n{category.upper().replace('_', ' ')}:")
            for rec in recs[:3]:  # Show top 3 recommendations per category
                print(f"  • {rec}")
    
    print(f"\nFor complete diagnostic results, check: {args.output}/")


if __name__ == "__main__":
    main()
