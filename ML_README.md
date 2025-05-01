# Dental Width Predictor: Machine Learning Implementation

This branch implements a machine learning-based approach for tooth detection and classification in dental radiographs. It provides an alternative to the traditional computer vision methods in the main branch.

## Overview

The ML implementation uses:

1. **U-Net Segmentation Model** - For segmenting teeth from radiographs and classifying them by type (primary molar vs premolar)
2. **CNN Classification Model** - For classifying individual tooth regions

This approach has several potential advantages:
- More robust to variations in image quality
- Better at handling complex anatomical variations
- Improved accuracy in detecting tooth boundaries
- Automatic classification of tooth types

## New Files

### Models

- `models/model.py` - Neural network model definitions and utility functions
- `src/training/train_models.py` - Training script for the segmentation and classification models

### ML-Based Detection and Classification

- `src/detection/ml_tooth_detection.py` - ML-based tooth detection functions
- `src/detection/ml_tooth_classification.py` - ML-based tooth classification functions
- `src/ml_main.py` - Entry point for ML-based processing

## Getting Started

### Prerequisites

Same as the main project, plus:
- TensorFlow 2.x (already in requirements.txt)

### Usage

#### Running Inference with Pre-trained Models

```bash
# Process a single image using ML models
python src/ml_main.py --image data/samples/sample1.jpg --output results/sample1_ml.jpg

# Process an image with debug visualizations
python src/ml_main.py --image data/samples/sample1.jpg --output results/sample1_ml.jpg --debug

# Use fallback to traditional methods if ML fails
python src/ml_main.py --image data/samples/sample1.jpg --output results/sample1_ml.jpg --fallback
```

#### Training Your Own Models

To train the models with your own data:

```bash
# Create a directory for your training data
mkdir -p data/training

# Organize your data (see Training Data Format below)
# ...

# Train both models
python src/training/train_models.py --data data/training --models models

# Train only the segmentation model
python src/training/train_models.py --data data/training --models models --segmentation_only

# Train only the classification model
python src/training/train_models.py --data data/training --models models --classification_only
```

### Training Data Format

For full training functionality, organize your data as follows:

```
data/training/
├── images/                 # Original radiograph images
├── segmentation_masks/     # Segmentation masks (pixel values: 0=background, 1=primary_molar, 2=premolar)
└── tooth_regions/          # Extracted tooth regions for classification
    ├── primary_molar/      # Examples of primary molars
    ├── premolar/           # Examples of premolars
    └── other/              # Examples of other teeth
```

## Implementation Details

### Segmentation Model (U-Net)

The segmentation model is based on the U-Net architecture, which is well-suited for medical image segmentation:

- **Input**: Dental radiograph (512x512 grayscale)
- **Output**: Multi-class segmentation mask (3 classes: background, primary molar, premolar)
- **Architecture**: 
  - Encoder: 4 blocks of dual conv + max pooling
  - Bridge: Dual conv with dropout
  - Decoder: 4 blocks of upsampling + skip connections + dual conv
  - Output: 1x1 conv with softmax activation

### Classification Model

The classification model is a standard CNN classifier:

- **Input**: Tooth region (128x128 grayscale)
- **Output**: Tooth type probabilities (primary molar, premolar, other)
- **Architecture**:
  - 4 convolutional blocks
  - Fully connected layers with dropout
  - Softmax output layer

## Comparison with Traditional Approach

The ML-based approach offers several advantages over the traditional computer vision methods:

1. **Robustness**: Better handles variations in image quality, orientation, and patient anatomy
2. **Accuracy**: Potentially higher accuracy in tooth segmentation and classification
3. **Adaptability**: Can learn from examples rather than requiring hand-crafted rules

However, it also has some limitations:

1. **Training Data**: Requires a dataset of labeled examples to train effectively
2. **Computational Requirements**: Needs more computational resources for inference
3. **Interpretability**: Less transparent in its decision-making process

## Future Improvements

Potential areas for further development:

1. **Active Learning**: Implement a feedback loop to continually improve the models
2. **Transfer Learning**: Adapt pre-trained dental models to improve performance with limited data
3. **Ensemble Methods**: Combine ML and traditional approaches for greater robustness
4. **Attention Mechanisms**: Incorporate attention for better focus on relevant features
5. **Explainability**: Add visualization of model attention to improve interpretability

## Integration with Main Branch

The ML implementation maintains compatibility with the original code:
- Uses the same preprocessing and measurement modules
- Provides fallback to traditional methods when needed
- Matches the original API for easier integration

## License

MIT