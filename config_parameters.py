#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration parameters for dental width predictor.

This module contains adjustable parameters for different aspects of the
detection and measurement pipeline. Users can modify these parameters
to improve results for their specific dataset.
"""

import cv2
import numpy as np

# Detection Parameters
DETECTION_CONFIG = {
    # Edge detection parameters (adjust based on image contrast)
    "edge_detection": {
        "sigma_values": [1.5, 2.5],  # Lower values for sharper edges, higher for smoother
        "sigma_alternatives": [
            [1.0, 3.0],  # For high contrast images
            [2.0, 3.5],  # For low contrast images
            [0.8, 4.0]   # For very detailed images
        ]
    },
    
    # Contour filtering parameters
    "contour_filtering": {
        "min_area": 200,        # Minimum area for tooth candidates (reduce for smaller teeth)
        "max_area": 25000,      # Maximum area for tooth candidates (increase for larger teeth)
        "min_eccentricity": 0.1,  # Minimum eccentricity (0 = circle, 1 = line)
        "min_solidity": 0.5,    # Minimum solidity (ratio of contour area to convex hull area)
        
        # Alternative parameter sets for different image types
        "alternative_sets": [
            {"min_area": 150, "max_area": 30000, "min_eccentricity": 0.08, "min_solidity": 0.4},
            {"min_area": 100, "max_area": 35000, "min_eccentricity": 0.12, "min_solidity": 0.6},
            {"min_area": 250, "max_area": 20000, "min_eccentricity": 0.15, "min_solidity": 0.45}
        ]
    },
    
    # Morphological operations parameters
    "morphology": {
        "dilation_radius": 2,    # Radius for edge dilation
        "closing_radius": 3,     # Radius for morphological closing
        "hole_fill_threshold": 50  # Minimum area for hole filling
    },
    
    # Dental arch filtering parameters
    "dental_arch": {
        "arch_top_ratio": 0.25,      # Top of dental arch (height fraction)
        "arch_bottom_ratio": 0.75,   # Bottom of dental arch (height fraction)
        "arch_width_ratio": 0.75,    # Width of dental arch (width fraction)
        "aspect_ratio_min": 0.5,     # Minimum aspect ratio for teeth
        "aspect_ratio_max": 3.5      # Maximum aspect ratio for teeth
    }
}

# Classification Parameters
CLASSIFICATION_CONFIG = {
    # Area-based classification thresholds
    "area_thresholds": {
        "large_tooth_multiplier": 1.2,    # Area multiplier for large teeth (primary molars)
        "small_tooth_multiplier": 0.8,    # Area multiplier for small teeth (premolars)
        "median_area_weight": 1.0          # Weight for median area calculation
    },
    
    # Shape-based classification
    "shape_features": {
        "compactness_threshold": 0.3,      # Minimum compactness for molars
        "texture_contrast_threshold": 50,  # Maximum texture contrast
        "texture_homogeneity_threshold": 0.5, # Minimum texture homogeneity
        "intensity_threshold": 100         # Minimum mean intensity
    },
    
    # Position-based classification
    "position_weights": {
        "upper_jaw_position_threshold": 0.7,  # Relative position in upper jaw
        "lower_jaw_position_threshold": 0.5,  # Relative position in lower jaw
        "horizontal_alignment_threshold": 75, # Maximum horizontal distance for pairing
        "vertical_separation_min": 30,        # Minimum vertical separation for pairs
        "vertical_separation_max": 150        # Maximum vertical separation for pairs
    }
}

# Measurement Parameters
MEASUREMENT_CONFIG = {
    # Contact point detection
    "contact_points": {
        "alignment_threshold": 75,         # Maximum horizontal distance for tooth pairing
        "vertical_separation_penalty": 2, # Penalty factor for vertical separation
        "pca_fallback_threshold": 1.2,    # Aspect ratio threshold for PCA vs simple measurement
    },
    
    # Width measurement methods
    "measurement_methods": {
        "primary_method": "pca_based",     # Options: "pca_based", "convex_hull", "bounding_box"
        "fallback_method": "convex_hull",  # Fallback if primary method fails
        "approximation_factor": 1.2       # Multiplier for bounding box approximations
    },
    
    # Calibration
    "calibration": {
        "default_factor": 0.1,            # Default mm/pixel ratio
        "auto_detect_dpi": True,          # Try to detect DPI from image metadata
        "common_dpi_values": [72, 96, 150, 300], # Common DPI values to try
        "pixel_size_estimates": {         # Rough estimates for different image sources
            "intraoral": 0.05,            # High resolution intraoral cameras
            "panoramic": 0.1,             # Standard panoramic radiographs
            "cbct": 0.2                   # CBCT slice images
        }
    }
}

# Preprocessing Parameters
PREPROCESSING_CONFIG = {
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    "clahe": {
        "clip_limit": 3.0,               # Contrast enhancement strength
        "tile_grid_size": (8, 8),        # Grid size for local enhancement
        
        # Adaptive CLAHE parameters based on image characteristics
        "adaptive_params": {
            "dark_image": {"clip_limit": 4.0, "tile_grid_size": (6, 6)},
            "bright_image": {"clip_limit": 2.0, "tile_grid_size": (10, 10)},
            "low_contrast": {"clip_limit": 5.0, "tile_grid_size": (8, 8)}
        }
    },
    
    # Filtering parameters
    "filtering": {
        "bilateral_d": 9,                # Bilateral filter neighborhood size
        "bilateral_sigma_color": 75,     # Color sigma for bilateral filter
        "bilateral_sigma_space": 75,     # Space sigma for bilateral filter
        "gaussian_kernel_size": (3, 3),  # Gaussian blur kernel size
        "gaussian_sigma": 0              # Gaussian blur sigma (0 = calculate from kernel)
    },
    
    # Histogram stretching
    "histogram": {
        "percentile_low": 2,             # Lower percentile for stretching
        "percentile_high": 98,           # Upper percentile for stretching
        "gamma_correction": 0.8          # Gamma value for correction
    },
    
    # Sharpening
    "sharpening": {
        "enable": True,                  # Enable sharpening
        "kernel": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], # Sharpening kernel
        "weight": 1.0                    # Weight for sharpening
    }
}

# Debug and Visualization Parameters
DEBUG_CONFIG = {
    # Output settings
    "save_intermediate": True,           # Save intermediate processing steps
    "save_failed_cases": True,          # Save results even for failed cases
    "verbose_logging": True,            # Enable detailed logging
    
    # Visualization settings
    "colors": {
        "primary_molar": (0, 255, 0),   # Green for primary molars
        "premolar": (0, 0, 255),        # Red for premolars
        "other": (255, 0, 0),           # Blue for other teeth
        "contour": (0, 255, 255),       # Yellow for contours
        "measurement": (255, 255, 0)    # Cyan for measurements
    },
    
    # Text settings
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.5,
    "font_thickness": 1,
    "text_color": (255, 255, 255)
}

# Batch Processing Parameters
BATCH_CONFIG = {
    # Processing limits
    "max_consecutive_failures": 10,     # Stop after this many consecutive failures
    "max_processing_time": 300,         # Maximum time per image (seconds)
    "chunk_size": 50,                   # Process images in chunks
    
    # Retry logic
    "retry_failed": True,               # Retry failed images with different parameters
    "retry_max_attempts": 3,            # Maximum retry attempts
    "retry_parameter_variations": [     # Different parameter sets for retries
        {"method": "traditional", "preprocessing": "enhanced"},
        {"method": "traditional", "preprocessing": "aggressive"},
        {"method": "traditional", "preprocessing": "gentle"}
    ],
    
    # Output settings
    "save_summary": True,               # Save CSV summary
    "save_detailed_log": True,          # Save detailed processing log
    "save_batch_report": True           # Save batch processing report
}

# Quality Control Parameters
QUALITY_CONFIG = {
    # Measurement validation
    "validation": {
        "min_width_mm": 2.0,            # Minimum reasonable tooth width (mm)
        "max_width_mm": 15.0,           # Maximum reasonable tooth width (mm)
        "max_width_difference": 8.0,    # Maximum reasonable width difference (mm)
        "min_pair_distance": 20,        # Minimum distance between paired teeth (pixels)
        "max_pair_distance": 200        # Maximum distance between paired teeth (pixels)
    },
    
    # Outlier detection
    "outlier_detection": {
        "enable": True,                 # Enable outlier detection
        "std_threshold": 2.0,           # Standard deviations for outlier detection
        "min_samples": 3                # Minimum samples needed for outlier detection
    },
    
    # Quality scoring
    "quality_scoring": {
        "min_contours": 4,              # Minimum contours for good quality
        "min_classifications": 2,       # Minimum classifications for good quality
        "min_measurements": 1           # Minimum measurements for good quality
    }
}

def get_config(config_name):
    """Get a specific configuration dictionary.
    
    Args:
        config_name (str): Name of the configuration
        
    Returns:
        dict: Configuration parameters
    """
    configs = {
        "detection": DETECTION_CONFIG,
        "classification": CLASSIFICATION_CONFIG,
        "measurement": MEASUREMENT_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "debug": DEBUG_CONFIG,
        "batch": BATCH_CONFIG,
        "quality": QUALITY_CONFIG
    }
    
    return configs.get(config_name, {})

def create_custom_config(image_type="panoramic", quality="standard"):
    """Create a custom configuration based on image type and quality requirements.
    
    Args:
        image_type (str): Type of image ("panoramic", "intraoral", "cbct")
        quality (str): Quality setting ("fast", "standard", "high_quality")
        
    Returns:
        dict: Custom configuration parameters
    """
    base_config = {
        "detection": DETECTION_CONFIG.copy(),
        "classification": CLASSIFICATION_CONFIG.copy(),
        "measurement": MEASUREMENT_CONFIG.copy(),
        "preprocessing": PREPROCESSING_CONFIG.copy()
    }
    
    # Adjust for image type
    if image_type == "panoramic":
        base_config["measurement"]["calibration"]["default_factor"] = 0.1
        base_config["detection"]["contour_filtering"]["min_area"] = 200
    elif image_type == "intraoral":
        base_config["measurement"]["calibration"]["default_factor"] = 0.05
        base_config["detection"]["contour_filtering"]["min_area"] = 400
    elif image_type == "cbct":
        base_config["measurement"]["calibration"]["default_factor"] = 0.2
        base_config["detection"]["contour_filtering"]["min_area"] = 100
    
    # Adjust for quality setting
    if quality == "fast":
        base_config["detection"]["edge_detection"]["sigma_values"] = [2.0, 3.0]
        base_config["preprocessing"]["clahe"]["clip_limit"] = 2.0
    elif quality == "high_quality":
        base_config["detection"]["edge_detection"]["sigma_values"] = [1.0, 2.0, 3.0, 4.0]
        base_config["preprocessing"]["clahe"]["clip_limit"] = 4.0
    
    return base_config

def save_config_to_file(config, filepath):
    """Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration to save
        filepath (str): Path to save the configuration
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def load_config_from_file(filepath):
    """Load configuration from a JSON file.
    
    Args:
        filepath (str): Path to the configuration file
        
    Returns:
        dict: Loaded configuration
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)

# Example usage and configuration templates
if __name__ == "__main__":
    print("Dental Width Predictor Configuration Parameters")
    print("=" * 50)
    
    # Show available configurations
    configs = ["detection", "classification", "measurement", "preprocessing", "debug", "batch", "quality"]
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n{config_name.upper()} PARAMETERS:")
        print(f"  Available parameters: {len(config)}")
        
        # Show a few key parameters
        if config_name == "detection":
            print(f"  - Min area: {config['contour_filtering']['min_area']}")
            print(f"  - Edge sigma: {config['edge_detection']['sigma_values']}")
        elif config_name == "measurement":
            print(f"  - Default calibration: {config['calibration']['default_factor']} mm/pixel")
            print(f"  - Alignment threshold: {config['contact_points']['alignment_threshold']} pixels")
    
    print(f"\nExample custom configurations:")
    
    # Create example configurations
    panoramic_config = create_custom_config("panoramic", "standard")
    intraoral_config = create_custom_config("intraoral", "high_quality")
    
    print(f"- Panoramic radiographs: {panoramic_config['measurement']['calibration']['default_factor']} mm/pixel")
    print(f"- Intraoral images: {intraoral_config['measurement']['calibration']['default_factor']} mm/pixel")
    
    # Save example configuration
    save_config_to_file(panoramic_config, "config_panoramic.json")
    print(f"\nExample configuration saved to: config_panoramic.json")
