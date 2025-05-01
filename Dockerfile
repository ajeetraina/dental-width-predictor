FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Fix webbrowser requirement (it's a standard library, not a pip package)
RUN sed -i '/webbrowser>=0.10.0/d' requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/samples data/my_radiographs models results

# Expose the port that dashboard.py might use
EXPOSE 8000 8080

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Create entry point script
RUN echo '#!/bin/bash\n\
if [ "$1" = "dashboard" ]; then\n\
    # Run the dashboard\n\
    python src/dashboard.py --input $2 --results $3 --serve --port ${4:-8000}\n\
elif [ "$1" = "batch" ]; then\n\
    # Run batch processing\n\
    python src/batch_processing.py --input $2 --output $3 ${@:4}\n\
elif [ "$1" = "process" ]; then\n\
    # Process a single image\n\
    python src/main.py --image $2 --output $3 ${@:4}\n\
elif [ "$1" = "ml-process" ]; then\n\
    # Process a single image using ML model\n\
    python src/ml_main.py --image $2 --output $3 ${@:4}\n\
elif [ "$1" = "ml-train" ]; then\n\
    # Train the ML models\n\
    python src/training/train_models.py --data $2 --models $3 ${@:4}\n\
else\n\
    # Default: print usage\n\
    echo "Dental Width Predictor"\n\
    echo "---------------------"\n\
    echo "Usage:"\n\
    echo "  dashboard <input_dir> <results_dir> [port]  - Run the interactive dashboard"\n\
    echo "  batch <input_dir> <output_dir> [options]    - Process multiple images"\n\
    echo "  process <image_path> <output_path> [options] - Process a single image"\n\
    echo "  ml-process <image_path> <output_path> [options] - Process using ML models"\n\
    echo "  ml-train <data_dir> <models_dir> [options]  - Train ML models"\n\
    echo ""\n\
    echo "For more options, see the README.md or ML_README.md"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entry point
ENTRYPOINT ["/app/entrypoint.sh"]

# Set default command (shows help)
CMD ["help"]