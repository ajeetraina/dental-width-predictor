#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard module for Dental Width Predictor.

This module implements a web-based dashboard for visualizing 
dental width measurement results across multiple images.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import webbrowser
from threading import Timer

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_processing import process_single_image, batch_process_directory


def generate_summary_stats(results_dir):
    """Generate summary statistics from measurement results.
    
    Args:
        results_dir (str): Directory containing measurement results
        
    Returns:
        dict: Summary statistics
    """
    # Find all JSON result files
    json_files = glob.glob(os.path.join(results_dir, "*_measurements.json"))
    
    if not json_files:
        return None
    
    # Collect all measurement data
    all_measurements = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        image_name = Path(data["image"]).name
        
        for pair in data["tooth_pairs"]:
            measurement = {
                'image': image_name,
                'pair_index': pair['pair_index'],
                'position': pair['position'],
                'primary_molar_width': pair['primary_molar_width'],
                'premolar_width': pair['premolar_width'],
                'width_difference': pair['width_difference']
            }
            all_measurements.append(measurement)
    
    # Convert to DataFrame
    if not all_measurements:
        return None
        
    df = pd.DataFrame(all_measurements)
    
    # Calculate summary statistics
    summary = {
        'total_images': len(json_files),
        'total_measurements': len(df),
        'avg_primary_width': df['primary_molar_width'].mean(),
        'avg_premolar_width': df['premolar_width'].mean(),
        'avg_width_difference': df['width_difference'].mean(),
        'std_width_difference': df['width_difference'].std(),
        'min_width_difference': df['width_difference'].min(),
        'max_width_difference': df['width_difference'].max(),
        'by_position': df.groupby('position').agg({
            'primary_molar_width': 'mean',
            'premolar_width': 'mean',
            'width_difference': ['mean', 'std', 'count']
        }).to_dict(),
        'raw_data': df.to_dict(orient='records')
    }
    
    return summary


def create_histogram_plot(data):
    """Create a histogram of width differences.
    
    Args:
        data (list): List of width difference measurements
        
    Returns:
        str: Base64-encoded PNG image
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Width Differences')
    plt.xlabel('Width Difference (mm)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    mean_val = np.mean(data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_val:.2f} mm')
    
    # Add research expectation line (approximately 2mm)
    plt.axvline(2.0, color='green', linestyle='dotted', linewidth=2,
                label='Expected: 2.00 mm')
    
    plt.legend()
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=80)
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(image_png).decode('utf-8')


def create_scatter_plot(df):
    """Create a scatter plot of primary molar vs premolar widths.
    
    Args:
        df (pandas.DataFrame): DataFrame with measurement data
        
    Returns:
        str: Base64-encoded PNG image
    """
    plt.figure(figsize=(10, 6))
    
    # Color points by position
    positions = df['position'].unique()
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, position in enumerate(positions):
        pos_data = df[df['position'] == position]
        plt.scatter(pos_data['primary_molar_width'], pos_data['premolar_width'], 
                   color=colors[i % len(colors)], alpha=0.7, label=position)
    
    # Add diagonal line (y=x)
    min_val = min(df['primary_molar_width'].min(), df['premolar_width'].min())
    max_val = max(df['primary_molar_width'].max(), df['premolar_width'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Width')
    
    plt.title('Primary Molar Width vs Premolar Width')
    plt.xlabel('Primary Molar Width (mm)')
    plt.ylabel('Premolar Width (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=80)
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(image_png).decode('utf-8')


def create_bar_chart(df):
    """Create a bar chart of average width differences by position.
    
    Args:
        df (pandas.DataFrame): DataFrame with measurement data
        
    Returns:
        str: Base64-encoded PNG image
    """
    plt.figure(figsize=(10, 6))
    
    # Group by position
    position_stats = df.groupby('position').agg({
        'width_difference': ['mean', 'std', 'count']
    })
    
    # Extract mean values and standard deviations
    means = position_stats['width_difference']['mean']
    stds = position_stats['width_difference']['std']
    
    # Create bar chart
    positions = means.index
    x_pos = np.arange(len(positions))
    
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(x_pos, positions)
    plt.ylabel('Average Width Difference (mm)')
    plt.title('Width Difference by Tooth Position')
    plt.grid(True, alpha=0.3)
    
    # Add count labels
    for i, pos in enumerate(positions):
        count = position_stats['width_difference']['count'][pos]
        plt.text(i, means[pos] + stds[pos] + 0.1, f'n={count}', ha='center')
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=80)
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(image_png).decode('utf-8')


def generate_dashboard_html(summary, results_dir):
    """Generate an HTML dashboard from the summary statistics.
    
    Args:
        summary (dict): Summary statistics
        results_dir (str): Directory containing result visualizations
        
    Returns:
        str: HTML content
    """
    if not summary:
        return "<html><body><h1>No measurement data found</h1></body></html>"
    
    # Convert raw data to DataFrame for visualization
    df = pd.DataFrame(summary['raw_data'])
    
    # Generate plots
    histogram = create_histogram_plot(df['width_difference'])
    scatter = create_scatter_plot(df)
    bar_chart = create_bar_chart(df)
    
    # Find all visualization images
    vis_images = glob.glob(os.path.join(results_dir, "*_visualization.jpg"))
    vis_images.sort()
    
    # Create image gallery HTML
    image_gallery = ""
    for img_path in vis_images:
        img_name = os.path.basename(img_path)
        original_name = img_name.replace("_visualization.jpg", "")
        
        # Read image and convert to base64
        with open(img_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Add to gallery
        image_gallery += f"""
        <div class="col-md-6 mb-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title">{original_name}</h5>
                </div>
                <img src="data:image/jpeg;base64,{img_data}" class="card-img-top" alt="{original_name}">
                <div class="card-body">
                    <a href="#" class="btn btn-primary btn-sm" onclick="showImageDetails('{original_name}'); return false;">View Details</a>
                </div>
            </div>
        </div>
        """
    
    # Create data table
    data_rows = ""
    for i, row in enumerate(summary['raw_data']):
        data_rows += f"""
        <tr>
            <td>{row['image']}</td>
            <td>{row['position']}</td>
            <td>{row['primary_molar_width']:.2f}</td>
            <td>{row['premolar_width']:.2f}</td>
            <td>{row['width_difference']:.2f}</td>
        </tr>
        """
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dental Width Predictor Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding-top: 20px; padding-bottom: 40px; }}
        .card-img-top {{ height: 300px; object-fit: contain; }}
        .nav-pills .nav-link.active {{ background-color: #198754; }}
    </style>
</head>
<body>
    <div class="container">
        <header class="mb-4">
            <h1 class="text-center">Dental Width Predictor Dashboard</h1>
            <p class="text-center text-muted">Analysis of {summary['total_images']} images with {summary['total_measurements']} tooth pairs</p>
        </header>
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h4>Summary Statistics</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="card text-center mb-3">
                                    <div class="card-header bg-primary text-white">Average Primary Molar Width</div>
                                    <div class="card-body">
                                        <h3>{summary['avg_primary_width']:.2f} mm</h3>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center mb-3">
                                    <div class="card-header bg-primary text-white">Average Premolar Width</div>
                                    <div class="card-body">
                                        <h3>{summary['avg_premolar_width']:.2f} mm</h3>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center mb-3">
                                    <div class="card-header bg-success text-white">Average Width Difference</div>
                                    <div class="card-body">
                                        <h3>{summary['avg_width_difference']:.2f} mm</h3>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center mb-3">
                                    <div class="card-header bg-info text-white">Standard Deviation</div>
                                    <div class="card-body">
                                        <h3>{summary['std_width_difference']:.2f} mm</h3>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="pills-visualizations-tab" data-bs-toggle="pill" 
                        data-bs-target="#pills-visualizations" type="button" role="tab" 
                        aria-controls="pills-visualizations" aria-selected="true">Visualizations</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="pills-images-tab" data-bs-toggle="pill" 
                        data-bs-target="#pills-images" type="button" role="tab" 
                        aria-controls="pills-images" aria-selected="false">Image Gallery</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="pills-data-tab" data-bs-toggle="pill" 
                        data-bs-target="#pills-data" type="button" role="tab" 
                        aria-controls="pills-data" aria-selected="false">Raw Data</button>
            </li>
        </ul>
        
        <div class="tab-content" id="pills-tabContent">
            <div class="tab-pane fade show active" id="pills-visualizations" role="tabpanel" 
                 aria-labelledby="pills-visualizations-tab">
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Distribution of Width Differences</h5>
                            </div>
                            <div class="card-body">
                                <img src="data:image/png;base64,{histogram}" class="img-fluid" alt="Histogram">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Primary Molar vs Premolar Width</h5>
                            </div>
                            <div class="card-body">
                                <img src="data:image/png;base64,{scatter}" class="img-fluid" alt="Scatter Plot">
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>Width Difference by Tooth Position</h5>
                            </div>
                            <div class="card-body">
                                <img src="data:image/png;base64,{bar_chart}" class="img-fluid" alt="Bar Chart">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="tab-pane fade" id="pills-images" role="tabpanel" aria-labelledby="pills-images-tab">
                <div class="row">
                    {image_gallery}
                </div>
            </div>
            
            <div class="tab-pane fade" id="pills-data" role="tabpanel" aria-labelledby="pills-data-tab">
                <div class="card">
                    <div class="card-header">
                        <h5>Measurement Data</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Image</th>
                                        <th>Position</th>
                                        <th>Primary Molar Width (mm)</th>
                                        <th>Premolar Width (mm)</th>
                                        <th>Width Difference (mm)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Image Details Modal -->
        <div class="modal fade" id="imageDetailsModal" tabindex="-1" aria-labelledby="imageDetailsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="imageDetailsModalLabel">Image Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body" id="imageDetailsContent">
                        <!-- Content will be loaded dynamically -->
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="text-center mt-4 text-muted">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Dental Width Predictor</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function showImageDetails(imageName) {{
            // In a real application, this would load data from a JSON file
            // For demonstration, we'll create some dummy content
            
            // Find the image data
            const imageData = {json.dumps(summary['raw_data'])}.filter(item => item.image.startsWith(imageName));
            
            if (imageData.length === 0) {{
                alert("No details found for this image");
                return;
            }}
            
            // Create HTML content
            let modalContent = `
                <div class="card mb-3">
                    <div class="card-body">
                        <h5>Measurements for ${imageName}</h5>
                    </div>
                </div>
                
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Tooth Pair</th>
                            <th>Position</th>
                            <th>Primary Molar (mm)</th>
                            <th>Premolar (mm)</th>
                            <th>Difference (mm)</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            imageData.forEach(item => {{
                modalContent += `
                    <tr>
                        <td>Pair ${item.pair_index + 1}</td>
                        <td>${item.position}</td>
                        <td>${item.primary_molar_width.toFixed(2)}</td>
                        <td>${item.premolar_width.toFixed(2)}</td>
                        <td>${item.width_difference.toFixed(2)}</td>
                    </tr>
                `;
            }});
            
            modalContent += `
                    </tbody>
                </table>
            `;
            
            document.getElementById('imageDetailsContent').innerHTML = modalContent;
            new bootstrap.Modal(document.getElementById('imageDetailsModal')).show();
        }}
    </script>
</body>
</html>
"""
    
    return html


def open_browser(port=8000):
    """Open web browser to view the dashboard."""
    webbrowser.open(f'http://localhost:{port}')


def serve_dashboard(html_content, port=8000):
    """Serve the dashboard using a simple HTTP server."""
    import http.server
    import socketserver
    from io import BytesIO
    
    class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode())
    
    # Open browser after a short delay
    Timer(1.5, lambda: open_browser(port)).start()
    
    # Start server
    with socketserver.TCPServer(("", port), SimpleHTTPRequestHandler) as httpd:
        print(f"Serving dashboard at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()


def generate_and_save_dashboard(input_dir, results_dir):
    """Generate dashboard and save to HTML file.
    
    Args:
        input_dir (str): Directory containing input images
        results_dir (str): Directory containing result visualizations
        
    Returns:
        str: Path to the saved HTML file
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if results already exist, if not process the images
    json_files = glob.glob(os.path.join(results_dir, "*_measurements.json"))
    if not json_files:
        print("No measurement results found. Processing images...")
        batch_process_directory(input_dir, results_dir)
    
    # Generate summary statistics
    summary = generate_summary_stats(results_dir)
    
    # Generate dashboard HTML
    html_content = generate_dashboard_html(summary, results_dir)
    
    # Save HTML to file
    dashboard_path = os.path.join(results_dir, "dashboard.html")
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to {dashboard_path}")
    return dashboard_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Dental Width Predictor Dashboard')
    parser.add_argument('--input', type=str, default='data/samples',
                        help='Path to input images directory')
    parser.add_argument('--results', type=str, default='results',
                        help='Path to results directory')
    parser.add_argument('--serve', action='store_true',
                        help='Serve the dashboard using a built-in HTTP server')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to serve the dashboard (if --serve is specified)')
    
    args = parser.parse_args()
    
    # Generate dashboard
    dashboard_path = generate_and_save_dashboard(args.input, args.results)
    
    # Serve dashboard if requested
    if args.serve:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        serve_dashboard(html_content, args.port)
    else:
        # Open dashboard in browser
        webbrowser.open(f'file://{os.path.abspath(dashboard_path)}')


if __name__ == "__main__":
    main()
