"""
Color utilities for team classification and jersey detection
"""

import cv2
import numpy as np


def extract_dominant_color(image_crop: np.ndarray, color_space: str = "hsv",
                          num_colors: int = 1) -> tuple:
    """
    Extract dominant color from image crop.
    
    Args:
        image_crop: Image crop (BGR)
        color_space: Color space ('hsv', 'lab', 'rgb')
        num_colors: Number of dominant colors to return
    
    Returns:
        Tuple of dominant color(s)
    """
    if color_space == "hsv":
        img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    elif color_space == "lab":
        img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
    else:
        img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    
    # Calculate mean color
    mean_color = cv2.mean(img)[:3]
    return tuple(int(c) for c in mean_color)


def get_color_distance(color1: tuple, color2: tuple) -> float:
    """
    Calculate Euclidean distance between two colors.
    
    Args:
        color1: Color tuple
        color2: Color tuple
    
    Returns:
        Euclidean distance
    """
    return np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(color1, color2)))
