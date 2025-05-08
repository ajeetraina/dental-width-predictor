#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical growth assessment module for dental width predictor.

This module provides functions to assess premolar growth defects based on 
mesiodistal width differences between primary second molars and underlying 
second premolars, and generates appropriate clinical recommendations.
"""

import numpy as np

# Clinical reference values based on dental literature
# Reference ranges for normal width differences between primary second molars and second premolars
REFERENCE_RANGES = {
    "upper_left": {
        "normal_difference_range": (1.4, 2.2),  # mm
        "normal_ratio_range": (1.15, 1.35)      # primary_width/premolar_width
    },
    "upper_right": {
        "normal_difference_range": (1.4, 2.2),  # mm
        "normal_ratio_range": (1.15, 1.35)      # primary_width/premolar_width
    },
    "lower_left": {
        "normal_difference_range": (1.7, 2.5),  # mm
        "normal_ratio_range": (1.2, 1.4)        # primary_width/premolar_width
    },
    "lower_right": {
        "normal_difference_range": (1.7, 2.5),  # mm
        "normal_ratio_range": (1.2, 1.4)        # primary_width/premolar_width
    }
}

# Clinical severity thresholds
SEVERITY_THRESHOLDS = {
    "mild": 0.5,     # mm beyond normal range
    "moderate": 1.0, # mm beyond normal range
    "severe": 1.5    # mm beyond normal range
}

def assess_growth_defect(tooth_pair):
    """Assess premolar growth defect based on width difference.
    
    Args:
        tooth_pair (dict): Paired measurement data for primary molar and premolar
        
    Returns:
        dict: Assessment results including severity and recommendations
    """
    # Extract position and measurements
    position = tooth_pair["primary_molar"]["position"]
    primary_width = tooth_pair["primary_molar"]["measurement"]["width"]
    premolar_width = tooth_pair["premolar"]["measurement"]["width"]
    width_difference = tooth_pair["width_difference"]
    
    # Calculate ratio
    ratio = primary_width / premolar_width if premolar_width > 0 else 0
    
    # Get reference values for this position
    reference = REFERENCE_RANGES.get(position, REFERENCE_RANGES["upper_left"])
    normal_diff_min, normal_diff_max = reference["normal_difference_range"]
    normal_ratio_min, normal_ratio_max = reference["normal_ratio_range"]
    
    # Determine if there's a growth defect (either premolar too small or too large)
    defect_type = None
    severity = "normal"
    deviation = 0
    
    if width_difference > normal_diff_max:
        # Premolar too small relative to primary molar
        defect_type = "small_premolar"
        deviation = width_difference - normal_diff_max
    elif width_difference < normal_diff_min:
        # Premolar too large relative to primary molar
        defect_type = "large_premolar"
        deviation = normal_diff_min - width_difference
    
    # Determine severity based on deviation from normal range
    if deviation > 0:
        if deviation >= SEVERITY_THRESHOLDS["severe"]:
            severity = "severe"
        elif deviation >= SEVERITY_THRESHOLDS["moderate"]:
            severity = "moderate"
        elif deviation >= SEVERITY_THRESHOLDS["mild"]:
            severity = "mild"
    
    # Generate recommendations based on defect type and severity
    recommendations = get_recommendations(defect_type, severity, width_difference, position, 
                                         primary_width, premolar_width)
    
    return {
        "position": position,
        "primary_width": primary_width,
        "premolar_width": premolar_width,
        "width_difference": width_difference,
        "ratio": ratio,
        "normal_difference_range": reference["normal_difference_range"],
        "normal_ratio_range": reference["normal_ratio_range"],
        "defect_type": defect_type,
        "severity": severity,
        "deviation": deviation,
        "recommendations": recommendations
    }

def get_recommendations(defect_type, severity, width_difference, position, primary_width, premolar_width):
    """Generate clinical recommendations based on defect type and severity.
    
    Args:
        defect_type (str): Type of growth defect (small_premolar, large_premolar, or None)
        severity (str): Severity of the defect (normal, mild, moderate, severe)
        width_difference (float): Width difference in mm
        position (str): Position in the dental arch
        primary_width (float): Width of primary molar in mm
        premolar_width (float): Width of premolar in mm
        
    Returns:
        list: List of clinical recommendations
    """
    recommendations = []
    
    if defect_type is None or severity == "normal":
        recommendations.append("No intervention needed. Growth pattern appears normal.")
        return recommendations
    
    # Common monitoring recommendation
    recommendations.append(f"Regular monitoring of tooth development with follow-up radiographs every 6 months.")
    
    if defect_type == "small_premolar":
        if severity == "mild":
            recommendations.extend([
                "Possible space maintenance to preserve adequate room for premolar eruption.",
                "Monitor for potential crowding in the area.",
                "Consider preventive measures to maintain primary molar for appropriate length of time."
            ])
        elif severity == "moderate":
            recommendations.extend([
                "Space maintenance recommended to ensure adequate eruption space.",
                "Consider orthodontic consultation for early intervention planning.",
                "Evaluate adjacent teeth positions and overall arch space.",
                "Monitor for potential impaction risk."
            ])
        elif severity == "severe":
            recommendations.extend([
                "Immediate orthodontic consultation recommended.",
                "Space maintenance critical to prevent impaction or ectopic eruption.",
                "Consider radiographic follow-up at 3-month intervals.",
                "Possible extraction of primary molar may be needed with appropriate space maintenance.",
                "Comprehensive treatment planning for potential space management issues."
            ])
    
    elif defect_type == "large_premolar":
        if severity == "mild":
            recommendations.extend([
                "Monitor for early primary molar loss due to premolar eruption pressure.",
                "Evaluate for potential spacing in the arch.",
                "Consider preventive measures to maintain primary molar as long as possible."
            ])
        elif severity == "moderate":
            recommendations.extend([
                "Orthodontic consultation recommended.",
                "Evaluate potential arch length discrepancies.",
                "Monitor for accelerated resorption of primary molar roots.",
                "Consider interproximal reduction of adjacent teeth if crowding is anticipated."
            ])
        elif severity == "severe":
            recommendations.extend([
                "Immediate orthodontic consultation recommended.",
                "Careful monitoring for ectopic eruption pathway.",
                "Consider staged extraction protocol if indicated.",
                "Comprehensive space analysis needed for treatment planning.",
                "May require early intervention to manage eruption path."
            ])
    
    # Add specific recommendation about the measurement
    if defect_type == "small_premolar":
        recommendations.append(f"Premolar is approximately {width_difference:.1f}mm smaller than expected " +
                              f"based on primary molar width of {primary_width:.1f}mm.")
    elif defect_type == "large_premolar":
        expected_diff = REFERENCE_RANGES[position]["normal_difference_range"][0]
        recommendations.append(f"Premolar is approximately {expected_diff - width_difference:.1f}mm larger than expected " +
                              f"based on primary molar width of {primary_width:.1f}mm.")
    
    return recommendations

def analyze_all_pairs(tooth_pairs):
    """Analyze all tooth pairs and provide comprehensive assessment.
    
    Args:
        tooth_pairs (list): List of tooth pair measurements
        
    Returns:
        dict: Comprehensive assessment results
    """
    if not tooth_pairs:
        return {
            "analyzed_pairs": [],
            "overall_assessment": "No valid tooth pairs found for analysis.",
            "needs_intervention": False
        }
    
    analyzed_pairs = []
    intervention_needed = False
    highest_severity = "normal"
    severity_scores = {"normal": 0, "mild": 0, "moderate": 0, "severe": 0}
    
    for pair in tooth_pairs:
        assessment = assess_growth_defect(pair)
        analyzed_pairs.append(assessment)
        
        # Update severity counts
        severity = assessment["severity"]
        severity_scores[severity] += 1
        
        # Update highest severity
        if severity == "severe":
            highest_severity = "severe"
            intervention_needed = True
        elif severity == "moderate" and highest_severity != "severe":
            highest_severity = "moderate"
            intervention_needed = True
        elif severity == "mild" and highest_severity not in ["severe", "moderate"]:
            highest_severity = "mild"
    
    # Generate overall assessment
    overall_assessment = generate_overall_assessment(severity_scores, analyzed_pairs)
    
    return {
        "analyzed_pairs": analyzed_pairs,
        "severity_counts": severity_scores,
        "highest_severity": highest_severity,
        "overall_assessment": overall_assessment,
        "needs_intervention": intervention_needed
    }

def generate_overall_assessment(severity_scores, analyzed_pairs):
    """Generate an overall assessment based on severity scores.
    
    Args:
        severity_scores (dict): Counts of different severity levels
        analyzed_pairs (list): List of analyzed tooth pairs
        
    Returns:
        str: Overall assessment message
    """
    total_pairs = sum(severity_scores.values())
    if total_pairs == 0:
        return "No valid tooth pairs found for analysis."
    
    if severity_scores["severe"] > 0:
        return f"Immediate orthodontic intervention recommended. {severity_scores['severe']} tooth pair(s) show severe growth discrepancy."
    
    if severity_scores["moderate"] > 0:
        return f"Orthodontic consultation recommended. {severity_scores['moderate']} tooth pair(s) show moderate growth discrepancy."
    
    if severity_scores["mild"] > 0:
        return f"Monitoring recommended. {severity_scores['mild']} tooth pair(s) show mild growth discrepancy."
    
    return "All tooth pairs show normal growth patterns. Routine monitoring recommended."
