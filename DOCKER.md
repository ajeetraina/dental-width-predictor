# Containerized Dental Width Predictor

This document explains how to use the containerized version of the Dental Width Predictor application.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional but recommended)

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/ajeetraina/dental-width-predictor.git
   cd dental-width-predictor
   ```

2. Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

   This will start the dashboard application and expose it on port 8000.

3. Access the dashboard by opening a web browser and navigating to:
   ```
   http://localhost:8000
   ```

### Using Docker Directly

1. Clone the repository:
   ```bash
   git clone https://github.com/ajeetraina/dental-width-predictor.git
   cd dental-width-predictor
   ```

2. Build the Docker image:
   ```bash
   docker build -t dental-width-predictor .
   ```

3. Run the container:
   ```bash
   docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results dental-width-predictor dashboard data/samples results 8000
   ```

4. Access the dashboard by opening a web browser and navigating to:
   ```
   http://localhost:8000
   ```

## Usage Modes

The containerized application supports three primary modes of operation:

### 1. Dashboard Mode

The dashboard provides an interactive web interface to visualize the measurements and statistics.

With Docker Compose:
```bash
docker-compose up
```

With Docker:
```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results dental-width-predictor dashboard data/samples results 8000
```

Parameters:
- `dashboard`: The operation mode
- `data/samples`: Input directory containing radiograph images
- `results`: Output directory for processed results
- `8000`: Port number to serve the dashboard (optional, default: 8000)

### 2. Batch Processing Mode

Process multiple radiograph images in a directory.

With Docker Compose:
```bash
docker-compose run --rm dental-predictor batch data/my_radiographs results
```

With Docker:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results dental-width-predictor batch data/my_radiographs results
```

Parameters:
- `batch`: The operation mode
- `data/my_radiographs`: Input directory containing radiograph images
- `results`: Output directory for processed results

### 3. Single Image Processing Mode

Process a single radiograph image.

With Docker Compose:
```bash
docker-compose run --rm dental-predictor process data/samples/sample1.jpg results/output.jpg
```

With Docker:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results dental-width-predictor process data/samples/sample1.jpg results/output.jpg
```

Parameters:
- `process`: The operation mode
- `data/samples/sample1.jpg`: Path to the input radiograph image
- `results/output.jpg`: Path to save the output visualization

## Working with Your Own Data

1. Place your radiograph images in the `data/my_radiographs` directory:
   ```bash
   mkdir -p data/my_radiographs
   cp /path/to/your/images/*.jpg data/my_radiographs/
   ```

2. Process your images in batch mode:
   ```bash
   docker-compose run --rm dental-predictor batch data/my_radiographs results
   ```

3. Generate and view the dashboard:
   ```bash
   docker-compose run -p 8000:8000 --rm dental-predictor dashboard data/my_radiographs results 8000
   ```

## Customizing the Container

### Environment Variables

You can customize the behavior by setting environment variables in the `docker-compose.yml` file:

```yaml
services:
  dental-predictor:
    environment:
      - DISPLAY=${DISPLAY}  # Required for displaying windows on Linux
      # Add custom environment variables here
```

### Volume Mounts

The container uses two main volume mounts:

1. `/app/data`: For input images
2. `/app/results`: For output results

You can customize these mounts in the `docker-compose.yml` file.

## Troubleshooting

### Dashboard Not Accessible

If you can't access the dashboard, check:

1. The container is running:
   ```bash
   docker ps
   ```

2. The correct port is exposed:
   ```bash
   docker-compose ps
   ```

3. Try accessing the dashboard with the IP address instead of localhost:
   ```
   http://127.0.0.1:8000
   ```

### Permission Issues with Mounted Volumes

If you encounter permission issues with the mounted volumes, run:

```bash
sudo chown -R $(id -u):$(id -g) data results
```

This will ensure that the current user has the correct permissions.

## Security Considerations

When using this container in a production environment:

1. Don't expose the dashboard to the public internet without proper authentication
2. Use non-root users inside the container
3. Consider adding TLS/SSL for secure connections

## License

MIT