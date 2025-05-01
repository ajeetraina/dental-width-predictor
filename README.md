# Dental Width Predictor

A tool for measuring and predicting tooth width differences between primary second molars and second premolars in dental radiographs.

## Overview

This project automates the process of measuring the width difference between primary second molars and underlying second premolars in dental panoramic radiographs. This measurement is valuable for orthodontic treatment planning and prediction of tooth development.

## Dental Terminology

Understanding dental terminology is important for using this tool:

```mermaid
graph TD
    subgraph Dental Arch
        Midline[Midline] --- UR["Upper Right Quadrant"]
        Midline --- UL["Upper Left Quadrant"]
        UR --- LR["Lower Right Quadrant"]
        UL --- LL["Lower Left Quadrant"]
    end
    
    subgraph "Mesiodistal Width"
        Tooth[Tooth Crown] --- M["Mesial Surface<br>(toward midline)"]
        Tooth --- D["Distal Surface<br>(away from midline)"]
        M <--"Mesiodistal Width<br>(measured at contact points)"--> D
    end
    
    subgraph "Primary vs Permanent Teeth"
        Primary["Primary Second Molar<br>(deciduous/baby tooth)"] --- Permanent["Second Premolar<br>(permanent/adult tooth)"]
        Difference["Width Difference = Primary Molar Width - Premolar Width"]
    end
    
    subgraph "Tooth Surfaces"
        T["Single Tooth"] --- MS["Mesial<br>(toward midline)"]
        T --- DS["Distal<br>(away from midline)"]
        T --- BS["Buccal<br>(toward cheek)"]
        T --- LS["Lingual<br>(toward tongue)"]
        T --- OS["Occlusal<br>(chewing surface)"]
        
        MS --- CP1["Contact Point"]
        DS --- CP2["Contact Point"]
        
        CP1 <--"Width Measurement"--> CP2
    end
    
    style Midline fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#bbf,stroke:#333,stroke-width:1px
    style D fill:#bbf,stroke:#333,stroke-width:1px
    style CP1 fill:#f99,stroke:#333,stroke-width:2px
    style CP2 fill:#f99,stroke:#333,stroke-width:2px
    style Primary fill:#9f9,stroke:#333,stroke-width:1px
    style Permanent fill:#9cf,stroke:#333,stroke-width:1px
    style Difference fill:#ff9,stroke:#333,stroke-width:1px
```

## Features

- Image preprocessing for dental radiographs
- Automatic detection of primary second molars and underlying second premolars
- Measurement of tooth width at the widest point (contact points)
- Calculation of width differences
- Visualization of measurements
- Batch processing for multiple images

## System Architecture

```mermaid
graph TD
    subgraph Project ["Dental Width Predictor Project"]
        Input[Dental Radiograph Image] --> Preproc
        
        subgraph Process ["Processing Pipeline"]
            Preproc[Image Preprocessing] --> Detection
            Detection[Tooth Detection] --> Classification
            Classification[Tooth Classification] --> Measurement
            Measurement[Width Measurement] --> Results
            Results[Width Difference Calculation] --> Visualization
        end
        
        Visualization[Results Visualization] --> Output
        Output[Annotated Image with Measurements]
    end
    
    subgraph Modules ["Software Modules"]
        ImProc[src/preprocessing/image_processing.py]
        ToothDet[src/detection/tooth_detection.py]
        ToothClass[src/detection/tooth_classification.py]
        WidthMeas[src/measurement/width_measurement.py]
        Vis[src/utils/visualization.py]
        Cal[src/utils/calibration.py]
        Main[src/main.py]
    end
    
    ImProc --> Preproc
    ToothDet --> Detection
    ToothClass --> Classification
    WidthMeas --> Measurement
    Cal --> Measurement
    Vis --> Visualization
    Main --> Process

classDef module fill:#f9f,stroke:#333,stroke-width:2px;
classDef process fill:#bbf,stroke:#333,stroke-width:1px;
classDef data fill:#ffa,stroke:#333,stroke-width:1px;

class ImProc,ToothDet,ToothClass,WidthMeas,Vis,Cal,Main module;
class Preproc,Detection,Classification,Measurement,Results,Visualization process;
class Input,Output data;
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ajeetraina/dental-width-predictor.git
cd dental-width-predictor

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Processing a Single Image

```bash
python src/main.py --image path/to/radiograph.jpg --output results/output.jpg
```

### Processing Multiple Images (Batch Processing)

For datasets with multiple images (20-30 images):

```bash
# Place your images in the data directory
mkdir -p data/my_radiographs
# Copy your images to data/my_radiographs/

# Run batch processing
python src/batch_processing.py --input data/my_radiographs --output results
```

This will:
1. Process each image in the directory
2. Save visualizations showing measurements
3. Save detailed measurement data as JSON files
4. Generate a CSV summary of all measurements

### Analyzing Results

After batch processing, you'll find in the `results` directory:
- `*_visualization.jpg`: Visual representation of detected teeth and measurements
- `*_measurements.json`: Detailed measurement data in JSON format
- `measurements_summary.csv`: Combined data from all images for statistical analysis

## Dataset Management

### Using Sample Images

The repository includes a `data/samples` directory where you can find example radiographs:

```bash
# Process a sample image
python src/main.py --image data/samples/sample1.jpg
```

### Adding Your Own Dataset

You have several options for working with your dataset:

#### Option 1: Add Small Sample Images to Git

For a few representative images (recommended for public repositories):

```bash
# Copy a few small sample images (anonymized) to the samples directory
cp path/to/anonymized_sample1.jpg data/samples/

# Add to Git repository
git add data/samples/*.jpg
git commit -m "Add anonymized sample radiographs"
git push
```

#### Option 2: Use Git LFS for Larger Datasets

For larger image sets (20-30 images), consider using [Git Large File Storage (LFS)](https://git-lfs.github.com/):

```bash
# Install Git LFS
git lfs install

# Track image files with Git LFS
git lfs track "*.jpg" "*.png" "*.tiff"
git add .gitattributes

# Create dataset directory
mkdir -p data/full_dataset

# Add images to the dataset directory
cp path/to/images/*.jpg data/full_dataset/

# Commit and push
git add data/full_dataset
git commit -m "Add full radiograph dataset using Git LFS"
git push
```

#### Option 3: Local Dataset (Not in Git)

For private datasets or very large files:

```bash
# Create a directory for your dataset (not tracked by Git)
mkdir -p data/my_radiographs

# Copy your images to this directory
cp path/to/images/*.jpg data/my_radiographs/

# Ensure the directory is ignored in .gitignore (already configured)
```

## Project Structure

```
dental-width-predictor/
├── data/               # Sample radiograph images and datasets
│   ├── samples/        # Example radiographs included in the repository
│   └── my_radiographs/ # Your dataset (not tracked by Git)
├── models/             # Pre-trained models for tooth detection
├── notebooks/          # Jupyter notebooks for visualization and testing
├── src/                # Source code
│   ├── preprocessing/  # Image preprocessing modules
│   ├── detection/      # Tooth detection algorithms
│   ├── measurement/    # Width measurement tools
│   ├── utils/          # Utility functions
│   ├── batch_processing.py # Module for processing multiple images
│   └── main.py         # Main entry point
├── tests/              # Unit tests
└── requirements.txt    # Dependencies
```

## How It Works

1. **Preprocessing**: Enhance the dental radiograph for better feature detection
2. **Tooth Detection**: Identify and segment individual teeth in the image
3. **Tooth Classification**: Classify and locate primary second molars and second premolars
4. **Width Measurement**: Measure the width at the widest points (contact points)
5. **Difference Calculation**: Calculate the width difference between corresponding teeth
6. **Visualization**: Display results with overlays showing measurements

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
