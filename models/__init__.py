# Dental Width Predictor Models Package
# This package contains the core AI models for tooth detection and measurement

from .model import (
    create_segmentation_model,
    create_tooth_classifier, 
    extract_tooth_measurements,
    create_cnn_classifier_model,
    create_unet_segmentation_model,
    preprocess_dental_xray,
    detect_calibration_markers
)

__version__ = "1.0.0"
__author__ = "Ajeet Singh Raina"
