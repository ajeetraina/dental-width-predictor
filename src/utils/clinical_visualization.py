#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical visualization module for dental width predictor.

This module provides enhanced visualization functions that highlight
clinical findings and growth concerns in dental radiographs.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

def create_clinical_visualization(image, tooth_pairs, clinical_analyses, output_path=None, 
                                  show_legend=True, show_measurements=True, 
                                  highlight_discrepancies=True):
    """Create an enhanced clinical visualization with growth assessments highlighted.
    
    Args:
        image (numpy.ndarray): Original radiograph image
        tooth_pairs (list): List of tooth pair measurements
        clinical_analyses (list): List of clinical analyses for each tooth pair
        output_path (str, optional): Path to save the visualization
        show_legend (bool): Whether to show a legend
        show_measurements (bool): Whether to show numerical measurements
        highlight_discrepancies (bool): Whether to highlight growth discrepancies
        
    Returns:
        numpy.ndarray: The visualization image
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Convert grayscale to RGB if needed
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Define colors based on severity
    severity_colors = {
        "normal": (0, 255, 0),    # Green
        "mild": (0, 255, 255),    # Yellow
        "moderate": (0, 128, 255), # Orange
        "severe": (0, 0, 255)     # Red
    }
    
    # Draw measurements and assessments for each tooth pair
    for i, (pair, analysis) in enumerate(zip(tooth_pairs, clinical_analyses)):
        # Extract primary molar data
        primary_molar = pair["primary_molar"]
        primary_contour = primary_molar.get("contour", None)
        primary_centroid = primary_molar.get("centroid", None)
        primary_position = primary_molar.get("position", "unknown")
        
        # Extract premolar data
        premolar = pair["premolar"]
        premolar_contour = premolar.get("contour", None)
        premolar_centroid = premolar.get("centroid", None)
        
        # Get contact points if available
        primary_contact_points = primary_molar.get("measurement", {}).get("contact_points", None)
        premolar_contact_points = premolar.get("measurement", {}).get("contact_points", None)
        
        # Get clinical assessment info
        severity = analysis.get("severity", "normal")
        defect_type = analysis.get("defect_type", None)
        
        # Determine color based on severity
        color = severity_colors.get(severity, (0, 255, 0))
        
        # Draw contours if available
        if primary_contour is not None:
            cv2.drawContours(vis_image, [primary_contour], -1, color, 2)
        if premolar_contour is not None:
            cv2.drawContours(vis_image, [premolar_contour], -1, color, 2)
        
        # Draw contact points and measurement lines
        if primary_contact_points and premolar_contact_points and show_measurements:
            # Draw primary molar contact points and measurement line
            p1, p2 = primary_contact_points
            cv2.circle(vis_image, p1, 3, color, -1)
            cv2.circle(vis_image, p2, 3, color, -1)
            cv2.line(vis_image, p1, p2, color, 1)
            
            # Draw premolar contact points and measurement line
            p3, p4 = premolar_contact_points
            cv2.circle(vis_image, p3, 3, color, -1)
            cv2.circle(vis_image, p4, 3, color, -1)
            cv2.line(vis_image, p3, p4, color, 1)
            
            # Calculate midpoints for label placement
            primary_midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            premolar_midpoint = ((p3[0] + p4[0]) // 2, (p3[1] + p4[1]) // 2)
            
            # Draw width measurements
            primary_width = pair["primary_molar"]["measurement"]["width"]
            premolar_width = pair["premolar"]["measurement"]["width"]
            diff = pair["width_difference"]
            
            cv2.putText(vis_image, f"{primary_width:.1f}mm", 
                       (primary_midpoint[0] + 5, primary_midpoint[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            cv2.putText(vis_image, f"{premolar_width:.1f}mm", 
                       (premolar_midpoint[0] + 5, premolar_midpoint[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Add tooth identifiers and severity indicators
        if primary_centroid and show_measurements:
            identifier = f"PM{i+1}"  # Primary Molar identifier
            cv2.putText(vis_image, identifier, 
                       (int(primary_centroid[0]) - 15, int(primary_centroid[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        if premolar_centroid and show_measurements:
            identifier = f"P{i+1}"  # Premolar identifier
            cv2.putText(vis_image, identifier, 
                       (int(premolar_centroid[0]) - 15, int(premolar_centroid[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # For non-normal cases, add a severity indicator
        if defect_type and highlight_discrepancies and primary_centroid and premolar_centroid:
            # Calculate position for indicator (between the teeth)
            indicator_x = (int(primary_centroid[0]) + int(premolar_centroid[0])) // 2
            indicator_y = (int(primary_centroid[1]) + int(premolar_centroid[1])) // 2
            
            # Draw attention circle for moderate and severe cases
            if severity in ["moderate", "severe"]:
                cv2.circle(vis_image, (indicator_x, indicator_y), 20, color, 2)
                
                # Add arrows indicating the issue
                if defect_type == "small_premolar":
                    # Draw arrow pointing to premolar
                    cv2.arrowedLine(vis_image, 
                                   (indicator_x, indicator_y - 10),
                                   (int(premolar_centroid[0]), int(premolar_centroid[1]) - 10),
                                   color, 2, tipLength=0.3)
                elif defect_type == "large_premolar":
                    # Draw arrow pointing to primary molar
                    cv2.arrowedLine(vis_image, 
                                   (indicator_x, indicator_y - 10),
                                   (int(primary_centroid[0]), int(primary_centroid[1]) - 10),
                                   color, 2, tipLength=0.3)
    
    # Add legend if requested
    if show_legend:
        legend_height = 30
        legend_width = vis_image.shape[1]
        legend = np.ones((legend_height * 4, legend_width, 3), dtype=np.uint8) * 255
        
        # Draw severity legend
        y_offset = 20
        cv2.putText(legend, "SEVERITY LEGEND:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        for i, (severity, color) in enumerate(severity_colors.items()):
            y_pos = y_offset + 30 + i * 30
            cv2.rectangle(legend, (10, y_pos - 15), (30, y_pos + 5), color, -1)
            cv2.putText(legend, severity.upper(), (40, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Add clinical note
        clinical_note = "Note: This visualization highlights width differences between primary second molars"
        clinical_note2 = "and second premolars. Color coding indicates clinical significance of findings."
        cv2.putText(legend, clinical_note, (legend_width // 2, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(legend, clinical_note2, (legend_width // 2, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Combine image and legend
        vis_image_with_legend = np.vstack((vis_image, legend))
    else:
        vis_image_with_legend = vis_image
    
    # Save visualization if path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image_with_legend)
    
    return vis_image_with_legend

def generate_comparative_chart(clinical_analyses, output_path=None):
    """Generate a comparative chart of width measurements.
    
    Args:
        clinical_analyses (list): List of clinical analyses for tooth pairs
        output_path (str, optional): Path to save the chart image
        
    Returns:
        matplotlib.figure.Figure: The generated chart figure
    """
    # Prepare data for plotting
    positions = []
    primary_widths = []
    premolar_widths = []
    differences = []
    normal_min_diffs = []
    normal_max_diffs = []
    severities = []
    
    for analysis in clinical_analyses:
        positions.append(analysis["position"])
        primary_widths.append(analysis["primary_width"])
        premolar_widths.append(analysis["premolar_width"])
        differences.append(analysis["width_difference"])
        normal_min_diffs.append(analysis["normal_difference_range"][0])
        normal_max_diffs.append(analysis["normal_difference_range"][1])
        severities.append(analysis["severity"])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    
    # Plot 1: Width comparison
    bar_width = 0.35
    x = np.arange(len(positions))
    
    primary_bars = ax1.bar(x - bar_width/2, primary_widths, bar_width, label='Primary Second Molar')
    premolar_bars = ax1.bar(x + bar_width/2, premolar_widths, bar_width, label='Second Premolar')
    
    # Configure first plot
    ax1.set_title('Tooth Width Comparison')
    ax1.set_xlabel('Tooth Position')
    ax1.set_ylabel('Width (mm)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in primary_bars + premolar_bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot 2: Width difference vs normal range
    ax2.plot(x, differences, 'o-', linewidth=2, label='Measured Difference')
    ax2.fill_between(x, normal_min_diffs, normal_max_diffs, alpha=0.2, color='green', label='Normal Range')
    
    # Add points with color based on severity
    colors = {'normal': 'green', 'mild': 'yellow', 'moderate': 'orange', 'severe': 'red'}
    for i, (diff, severity) in enumerate(zip(differences, severities)):
        ax2.plot(i, diff, 'o', markersize=10, color=colors[severity])
    
    # Configure second plot
    ax2.set_title('Width Difference vs Normal Range')
    ax2.set_xlabel('Tooth Position')
    ax2.set_ylabel('Width Difference (mm)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels for differences
    for i, diff in enumerate(differences):
        ax2.annotate(f'{diff:.1f}',
                    xy=(i, diff),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center')
    
    # Add severity legend
    handles = [patches.Patch(color=color, label=severity.capitalize()) 
              for severity, color in colors.items()]
    legend2 = ax2.legend(handles=handles, title="Severity", loc='lower right')
    ax2.add_artist(legend2)
    
    plt.tight_layout()
    
    # Save chart if path is provided
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    
    return fig

def create_heatmap_visualization(image, tooth_pairs, clinical_analyses, output_path=None):
    """Create a heatmap visualization showing areas of concern.
    
    Args:
        image (numpy.ndarray): Original radiograph image
        tooth_pairs (list): List of tooth pair measurements
        clinical_analyses (list): List of clinical analyses for each tooth pair
        output_path (str, optional): Path to save the visualization
        
    Returns:
        numpy.ndarray: The heatmap visualization image
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Convert grayscale to RGB if needed
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Create a transparent overlay for the heatmap
    heatmap = np.zeros(vis_image.shape[:2], dtype=np.float32)
    
    # Define severity weights
    severity_weights = {
        "normal": 0.0,
        "mild": 0.3,
        "moderate": 0.6,
        "severe": 1.0
    }
    
    # Add heat based on clinical analyses
    for pair, analysis in zip(tooth_pairs, clinical_analyses):
        # Extract tooth data
        primary_molar = pair["primary_molar"]
        premolar = pair["premolar"]
        
        if "contour" not in primary_molar or "contour" not in premolar:
            continue
        
        # Get clinical assessment info
        severity = analysis.get("severity", "normal")
        weight = severity_weights.get(severity, 0.0)
        
        if weight > 0:
            # Create mask for relevant teeth
            mask = np.zeros(vis_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [primary_molar["contour"]], -1, 255, -1)
            cv2.drawContours(mask, [premolar["contour"]], -1, 255, -1)
            
            # Add weight to heatmap in teeth areas
            heatmap[mask > 0] = max(heatmap[mask > 0], weight)
    
    # Apply color mapping to heatmap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create mask for areas with heat
    heat_mask = (heatmap > 0).astype(np.uint8) * 255
    
    # Dilate heat mask to create smooth transitions
    kernel = np.ones((5, 5), np.uint8)
    heat_mask_dilated = cv2.dilate(heat_mask, kernel, iterations=3)
    
    # Create alpha channel for blending (higher values = more transparency)
    alpha = np.ones(vis_image.shape[:2], dtype=np.float32) - (heat_mask_dilated / 255.0) * 0.5
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    
    # Blend original image with heatmap
    blended = (vis_image * alpha + heatmap_color * (1 - alpha)).astype(np.uint8)
    
    # Add title and legend
    title_bar = np.ones((40, blended.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, "Growth Discrepancy Heatmap", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Create legend
    legend = np.ones((60, blended.shape[1], 3), dtype=np.uint8) * 255
    
    # Draw color gradient bar
    gradient_width = 200
    gradient_height = 20
    gradient_x = legend.shape[1] - gradient_width - 10
    gradient_y = 20
    
    for i in range(gradient_width):
        value = i / gradient_width
        color = cv2.applyColorMap(np.array([[int(value * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        cv2.rectangle(legend, (gradient_x + i, gradient_y), (gradient_x + i + 1, gradient_y + gradient_height), 
                     (int(color[0]), int(color[1]), int(color[2])), -1)
    
    # Add labels for legend
    cv2.putText(legend, "Normal", (gradient_x, gradient_y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(legend, "Severe", (gradient_x + gradient_width - 50, gradient_y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Combine image with title and legend
    vis_image_with_legend = np.vstack((title_bar, blended, legend))
    
    # Save visualization if path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image_with_legend)
    
    return vis_image_with_legend
