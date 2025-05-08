#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report generator module for dental width predictor.

This module provides functions to generate comprehensive reports including
measurements, visualizations, and clinical recommendations for dental
radiograph analysis.
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

def generate_clinical_report(image_path, measurements, analysis_results, output_dir, patient_info=None):
    """Generate a comprehensive clinical report with measurements and recommendations.
    
    Args:
        image_path (str): Path to the original radiograph
        measurements (list): Tooth measurement results
        analysis_results (dict): Growth assessment results
        output_dir (str): Directory to save the report
        patient_info (dict, optional): Patient information
        
    Returns:
        str: Path to the generated report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report filename
    base_name = Path(image_path).stem
    report_path = os.path.join(output_dir, f"{base_name}_clinical_report.json")
    
    # Prepare report data
    report_data = {
        "report_type": "Dental Width Predictor Clinical Report",
        "generated_date": datetime.now().isoformat(),
        "radiograph": os.path.basename(image_path),
        "patient_info": patient_info or {},
        "measurements": measurements,
        "analysis_results": analysis_results,
        "visualization_path": os.path.join(output_dir, f"{base_name}_visualization.jpg")
    }
    
    # Add summary section
    report_data["summary"] = {
        "total_pairs_analyzed": len(measurements),
        "pairs_with_defects": sum(1 for pair in analysis_results["analyzed_pairs"] if pair["defect_type"] is not None),
        "highest_severity": analysis_results["highest_severity"],
        "overall_assessment": analysis_results["overall_assessment"],
        "needs_intervention": analysis_results["needs_intervention"]
    }
    
    # Save report as JSON
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Also generate a human-readable text report
    text_report_path = os.path.join(output_dir, f"{base_name}_clinical_report.txt")
    _generate_text_report(report_data, text_report_path)
    
    return report_path

def _generate_text_report(report_data, output_path):
    """Generate a human-readable text report.
    
    Args:
        report_data (dict): Report data in JSON format
        output_path (str): Path to save the text report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{report_data['report_type']}\n")
        f.write("=" * 80 + "\n\n")
        
        # Date and radiograph info
        f.write(f"Date: {datetime.fromisoformat(report_data['generated_date']).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Radiograph: {report_data['radiograph']}\n\n")
        
        # Patient info if available
        if report_data['patient_info']:
            f.write("Patient Information:\n")
            f.write("-" * 20 + "\n")
            for key, value in report_data['patient_info'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
        
        # Summary section
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total tooth pairs analyzed: {report_data['summary']['total_pairs_analyzed']}\n")
        f.write(f"Pairs with growth discrepancies: {report_data['summary']['pairs_with_defects']}\n")
        f.write(f"Highest severity level: {report_data['summary']['highest_severity'].upper()}\n")
        f.write(f"Overall assessment: {report_data['summary']['overall_assessment']}\n\n")
        
        # Detailed analysis results
        f.write("DETAILED ANALYSIS\n")
        f.write("-" * 20 + "\n")
        
        for i, pair in enumerate(report_data['analysis_results']['analyzed_pairs']):
            f.write(f"Tooth Pair {i+1} ({pair['position']}):\n")
            f.write(f"  Primary Second Molar Width: {pair['primary_width']:.2f} mm\n")
            f.write(f"  Second Premolar Width: {pair['premolar_width']:.2f} mm\n")
            f.write(f"  Width Difference: {pair['width_difference']:.2f} mm\n")
            f.write(f"  Normal range for this position: {pair['normal_difference_range'][0]:.1f}-{pair['normal_difference_range'][1]:.1f} mm\n")
            
            if pair['defect_type'] is None:
                f.write("  Assessment: NORMAL growth pattern\n")
            else:
                defect_description = "Premolar too small" if pair['defect_type'] == "small_premolar" else "Premolar too large"
                f.write(f"  Assessment: {defect_description} ({pair['severity'].upper()} severity)\n")
                f.write(f"  Deviation from normal range: {pair['deviation']:.2f} mm\n")
                
                # Recommendations
                f.write("  Recommendations:\n")
                for rec in pair['recommendations']:
                    f.write(f"    ? {rec}\n")
            
            f.write("\n")
        
        # Closing
        f.write("=" * 80 + "\n")
        f.write("Note: This report is generated automatically and should be reviewed by a qualified\n")
        f.write("dental professional. The measurements and recommendations are based on\n")
        f.write("automated analysis and should be confirmed clinically.\n")
        f.write("=" * 80 + "\n")

def generate_batch_summary_csv(measurements_list, analysis_results_list, output_dir):
    """Generate a CSV summary for batch processing.
    
    Args:
        measurements_list (list): List of measurements for multiple radiographs
        analysis_results_list (list): List of analysis results
        output_dir (str): Directory to save the CSV file
        
    Returns:
        str: Path to the generated CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "clinical_measurements_summary.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            "Radiograph", "Position", "Primary Molar Width (mm)", 
            "Premolar Width (mm)", "Width Difference (mm)", 
            "Normal Range (mm)", "Defect Type", "Severity",
            "Deviation (mm)", "Needs Intervention"
        ])
        
        # Write data for each radiograph and tooth pair
        for i, (measurements, analysis) in enumerate(zip(measurements_list, analysis_results_list)):
            radiograph_name = measurements["image"] if "image" in measurements else f"Radiograph {i+1}"
            
            for pair_idx, pair_analysis in enumerate(analysis["analyzed_pairs"]):
                normal_range = f"{pair_analysis['normal_difference_range'][0]:.1f}-{pair_analysis['normal_difference_range'][1]:.1f}"
                defect_type = pair_analysis['defect_type'] if pair_analysis['defect_type'] else "normal"
                
                writer.writerow([
                    radiograph_name,
                    pair_analysis["position"],
                    f"{pair_analysis['primary_width']:.2f}",
                    f"{pair_analysis['premolar_width']:.2f}",
                    f"{pair_analysis['width_difference']:.2f}",
                    normal_range,
                    defect_type,
                    pair_analysis["severity"],
                    f"{pair_analysis['deviation']:.2f}" if pair_analysis['deviation'] > 0 else "0.00",
                    "Yes" if pair_analysis["severity"] in ["moderate", "severe"] else "No"
                ])
    
    return csv_path
