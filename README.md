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

```bash
python src/main.py --image path/to/radiograph.jpg
```

Or use the provided Jupyter notebooks in the `notebooks` directory for interactive analysis.

## Project Structure

```
dental-width-predictor/
├── data/               # Sample radiograph images and datasets
├── models/             # Pre-trained models for tooth detection
├── notebooks/          # Jupyter notebooks for visualization and testing
├── src/                # Source code
│   ├── preprocessing/  # Image preprocessing modules
│   ├── detection/      # Tooth detection algorithms
│   ├── measurement/    # Width measurement tools
│   ├── utils/          # Utility functions
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
