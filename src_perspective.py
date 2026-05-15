"""
Step 3: Perspective Transformation (Bird's Eye View)
Converts field-view player coordinates to top-down tactical map using homography
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class PerspectiveTransformer:
    """
    Performs perspective transformation from camera view to bird's-eye (top-down) view.
    Maps real-world pitch dimensions to 2D tactical map.
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 output_width: int = 1050, output_height: int = 680):
        """
        Initialize perspective transformer.
        
        Args:
            pitch_length: Real pitch length in meters (default: 105m)
            pitch_width: Real pitch width in meters (default: 68m)
            output_width: Output map width in pixels
            output_height: Output map height in pixels
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.output_width = output_width
        self.output_height = output_height
        
        self.transformation_matrix = None
        self.inverse_matrix = None
        self.tactical_map = None
        
        logger.info(f"PerspectiveTransformer initialized: {pitch_length}x{pitch_width}m, "
                   f"output {output_width}x{output_height}px")
    
    def set_source_points(self, src_points: List[Tuple[int, int]]) -> None:
        """
        Set the source points (camera view) - typically pitch corners.
        Points should be in order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        
        Args:
            src_points: List of 4 (x, y) tuples
        """
        if len(src_points) != 4:
            raise ValueError("Must provide exactly 4 source points")
        
        self.src_points = np.float32(src_points)
        logger.info(f"Source points set: {src_points}")
        self._compute_transformation_matrix()
    
    def _compute_transformation_matrix(self) -> None:
        """Compute perspective transformation matrix using source and destination points."""
        # Destination points (top-down view corners)
        dst_points = np.float32([
            [0, 0],
            [self.output_width, 0],
            [self.output_width, self.output_height],
            [0, self.output_height]
        ])
        
        # Calculate homography matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.src_points, dst_points
        )
        
        # Calculate inverse for mapping coordinates back
        self.inverse_matrix = cv2.getPerspectiveTransform(
            dst_points, self.src_points
        )
        
        logger.debug("Transformation matrices computed")
    
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to entire frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Transformed frame (bird's-eye view)
        """
        if self.transformation_matrix is None:
            raise RuntimeError("Source points not set. Call set_source_points() first.")
        
        warped = cv2.warpPerspective(
            frame,
            self.transformation_matrix,
            (self.output_width, self.output_height)
        )
        
        self.tactical_map = warped
        return warped
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from camera view to bird's-eye view.
        
        Args:
            points: Array of (x, y) coordinates in camera view
                   Shape: (N, 2) where N is number of points
        
        Returns:
            Transformed points in bird's-eye view
        """
        if self.transformation_matrix is None:
            raise RuntimeError("Source points not set. Call set_source_points() first.")
        
        # Reshape for perspectiveTransform
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply transformation
        transformed = cv2.perspectiveTransform(points, self.transformation_matrix)
        
        return transformed.reshape(-1, 2).astype(int)
    
    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from bird's-eye view back to camera view.
        
        Args:
            points: Array of (x, y) coordinates in bird's-eye view
        
        Returns:
            Points in original camera view
        """
        if self.inverse_matrix is None:
            raise RuntimeError("Inverse matrix not computed.")
        
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.inverse_matrix)
        
        return transformed.reshape(-1, 2).astype(int)
    
    def create_empty_tactical_map(self, bg_color: Tuple[int, int, int] = (50, 100, 50)) -> np.ndarray:
        """
        Create an empty tactical map with pitch markings.
        
        Args:
            bg_color: Background color (BGR)
        
        Returns:
            Empty tactical map with field markings
        """
        # Create green pitch background
        tactical_map = np.full(
            (self.output_height, self.output_width, 3),
            bg_color,
            dtype=np.uint8
        )
        
        # Draw pitch center line
        cv2.line(tactical_map,
                (self.output_width // 2, 0),
                (self.output_width // 2, self.output_height),
                (255, 255, 255), 2)
        
        # Draw center circle
        center = (self.output_width // 2, self.output_height // 2)
        radius = int(9.15 * self.output_width / self.pitch_length)  # 9.15m radius
        cv2.circle(tactical_map, center, radius, (255, 255, 255), 2)
        
        # Draw penalty areas
        penalty_width = int(40.32 * self.output_width / self.pitch_length)
        penalty_height = int(16.5 * self.output_height / self.pitch_width)
        
        # Home team penalty
        cv2.rectangle(tactical_map, (0, (self.output_height - penalty_height) // 2),
                     (penalty_width, (self.output_height + penalty_height) // 2),
                     (255, 255, 255), 2)
        
        # Away team penalty
        cv2.rectangle(tactical_map, (self.output_width - penalty_width, (self.output_height - penalty_height) // 2),
                     (self.output_width, (self.output_height + penalty_height) // 2),
                     (255, 255, 255), 2)
        
        self.tactical_map = tactical_map
        return tactical_map