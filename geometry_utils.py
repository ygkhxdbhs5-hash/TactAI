"""
Geometry utilities for tactical analysis
"""

import numpy as np
from typing import List, Tuple


def calculate_centroid(points: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Calculate centroid of a set of points.
    
    Args:
        points: List of (x, y) tuples
    
    Returns:
        Centroid (x, y)
    """
    if not points:
        return (0, 0)
    
    points = np.array(points)
    centroid = points.mean(axis=0)
    return tuple(centroid.astype(int))


def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: Point 1 (x, y)
        p2: Point 2 (x, y)
    
    Returns:
        Distance
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def point_in_circle(point: Tuple[int, int], center: Tuple[int, int],
                   radius: float) -> bool:
    """
    Check if point is inside circle.
    
    Args:
        point: Point (x, y)
        center: Circle center (x, y)
        radius: Circle radius
    
    Returns:
        True if point is inside circle
    """
    distance = calculate_distance(point, center)
    return distance <= radius
