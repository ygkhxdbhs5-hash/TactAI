"""
Step 4: Tactical Overlays (Visuals)
Draws team shapes, centroids, and ball-to-player lines on frames
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TacticalOverlay:
    """
    Draws tactical overlays on video frames for analysis visualization.
    Includes team convex hulls, centroids, and ball-to-nearest-player lines.
    """
    
    def __init__(self, line_thickness: int = 2, centroid_radius: int = 8,
                 hull_alpha: float = 0.3, text_scale: float = 0.6):
        """
        Initialize tactical overlay renderer.
        
        Args:
            line_thickness: Thickness for drawing lines
            centroid_radius: Radius for centroid circles
            hull_alpha: Alpha blending for convex hulls (0-1)
            text_scale: Font scale for text annotations
        """
        self.line_thickness = line_thickness
        self.centroid_radius = centroid_radius
        self.hull_alpha = hull_alpha
        self.text_scale = text_scale
        
        # Team colors (BGR)
        self.HOME_TEAM_COLOR = (0, 255, 0)     # Green
        self.AWAY_TEAM_COLOR = (255, 0, 0)     # Blue
        self.BALL_COLOR = (255, 255, 255)      # White
        self.HIGHLIGHT_COLOR = (0, 255, 255)   # Yellow
        
        logger.info("Tactical overlay initialized")
    
    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw bounding boxes for all detected players and ball.
        
        Args:
            frame: Input frame
            detections: Detection results with team classification
        
        Returns:
            Annotated frame
        """
        result = frame.copy()
        
        # Draw player bounding boxes
        for player in detections['players']:
            x1, y1 = player['x1'], player['y1']
            x2, y2 = player['x2'], player['y2']
            
            color = self.HOME_TEAM_COLOR if player['team'] == 0 else self.AWAY_TEAM_COLOR
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Draw player ID if available
            label = f"P#{player.get('track_id', '?')}"
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                       color, self.line_thickness)
        
        # Draw ball
        if detections['ball']:
            ball = detections['ball']
            cx, cy = ball['center_x'], ball['center_y']
            cv2.circle(result, (cx, cy), 5, self.BALL_COLOR, -1)
            cv2.circle(result, (cx, cy), 8, self.BALL_COLOR, self.line_thickness)
        
        return result
    
    def draw_team_shapes(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw convex hulls around each team to show team shape.
        
        Args:
            frame: Input frame
            detections: Detection results
        
        Returns:
            Annotated frame with team shapes
        """
        result = frame.copy()
        
        # Get player positions for each team
        home_positions = np.array([[p['center_x'], p['center_y']] 
                                  for p in detections['players'] if p['team'] == 0])
        away_positions = np.array([[p['center_x'], p['center_y']] 
                                  for p in detections['players'] if p['team'] == 1])
        
        # Draw home team convex hull
        if len(home_positions) >= 3:
            hull = cv2.convexHull(home_positions)
            
            # Draw filled hull with transparency
            overlay = result.copy()
            cv2.drawContours(overlay, [hull], 0, self.HOME_TEAM_COLOR, -1)
            result = cv2.addWeighted(overlay, self.hull_alpha, result, 
                                    1 - self.hull_alpha, 0)
            
            # Draw hull outline
            cv2.drawContours(result, [hull], 0, self.HOME_TEAM_COLOR, 
                           self.line_thickness)
        
        # Draw away team convex hull
        if len(away_positions) >= 3:
            hull = cv2.convexHull(away_positions)
            
            # Draw filled hull with transparency
            overlay = result.copy()
            cv2.drawContours(overlay, [hull], 0, self.AWAY_TEAM_COLOR, -1)
            result = cv2.addWeighted(overlay, self.hull_alpha, result, 
                                    1 - self.hull_alpha, 0)
            
            # Draw hull outline
            cv2.drawContours(result, [hull], 0, self.AWAY_TEAM_COLOR, 
                           self.line_thickness)
        
        return result
    
    def draw_centroids(self, frame: np.ndarray, team_stats: Dict) -> np.ndarray:
        """
        Draw centroids (center of mass) for each team.
        
        Args:
            frame: Input frame
            team_stats: Team statistics with centroid information
        
        Returns:
            Annotated frame with centroids
        """
        result = frame.copy()
        
        # Draw home team centroid
        if team_stats['home_team']['centroid']:
            cx, cy = team_stats['home_team']['centroid']
            cv2.circle(result, (int(cx), int(cy)), self.centroid_radius,
                      self.HOME_TEAM_COLOR, -1)
            cv2.circle(result, (int(cx), int(cy)), self.centroid_radius,
                      (255, 255, 255), self.line_thickness)
            cv2.putText(result, "HT", (int(cx) - 10, int(cy) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                       self.HOME_TEAM_COLOR, self.line_thickness)
        
        # Draw away team centroid
        if team_stats['away_team']['centroid']:
            cx, cy = team_stats['away_team']['centroid']
            cv2.circle(result, (int(cx), int(cy)), self.centroid_radius,
                      self.AWAY_TEAM_COLOR, -1)
            cv2.circle(result, (int(cx), int(cy)), self.centroid_radius,
                      (255, 255, 255), self.line_thickness)
            cv2.putText(result, "AT", (int(cx) - 10, int(cy) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                       self.AWAY_TEAM_COLOR, self.line_thickness)
        
        return result
    
    def draw_ball_to_nearest_player(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw a line from ball to the nearest player.
        
        Args:
            frame: Input frame
            detections: Detection results
        
        Returns:
            Annotated frame
        """
        result = frame.copy()
        
        if not detections['ball'] or not detections['players']:
            return result
        
        ball = detections['ball']
        ball_pos = np.array([ball['center_x'], ball['center_y']])
        
        # Find nearest player
        min_distance = float('inf')
        nearest_player = None
        
        for player in detections['players']:
            player_pos = np.array([player['center_x'], player['center_y']])
            distance = np.linalg.norm(ball_pos - player_pos)
            
            if distance < min_distance:
                min_distance = distance
                nearest_player = player
        
        if nearest_player:
            player_pos = (nearest_player['center_x'], nearest_player['center_y'])
            ball_pos_tuple = (ball['center_x'], ball['center_y'])
            
            # Draw line
            cv2.line(result, ball_pos_tuple, player_pos,
                    self.HIGHLIGHT_COLOR, self.line_thickness)
            
            # Draw distance label
            mid_x = (ball['center_x'] + nearest_player['center_x']) // 2
            mid_y = (ball['center_y'] + nearest_player['center_y']) // 2
            distance_text = f"{min_distance:.1f}px"
            cv2.putText(result, distance_text, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                       self.HIGHLIGHT_COLOR, self.line_thickness)
        
        return result
    
    def draw_on_tactical_map(self, tactical_map: np.ndarray, detections: Dict,
                            perspective_transformer) -> np.ndarray:
        """
        Draw detections and overlays on the tactical (bird's-eye) map.
        
        Args:
            tactical_map: Empty tactical map from perspective transformer
            detections: Detection results in camera view
            perspective_transformer: PerspectiveTransformer instance
        
        Returns:
            Annotated tactical map
        """
        result = tactical_map.copy()
        
        # Transform player positions
        for player in detections['players']:
            pos = np.array([[player['center_x'], player['center_y']]])
            transformed_pos = perspective_transformer.transform_points(pos)[0]
            
            color = self.HOME_TEAM_COLOR if player['team'] == 0 else self.AWAY_TEAM_COLOR
            
            cv2.circle(result, tuple(transformed_pos), 4, color, -1)
            cv2.circle(result, tuple(transformed_pos), 6, (255, 255, 255),
                      self.line_thickness)
        
        # Transform and draw ball
        if detections['ball']:
            ball = detections['ball']
            ball_pos = np.array([[ball['center_x'], ball['center_y']]])
            transformed_ball = perspective_transformer.transform_points(ball_pos)[0]
            
            cv2.circle(result, tuple(transformed_ball), 3, self.BALL_COLOR, -1)
            cv2.circle(result, tuple(transformed_ball), 5, self.BALL_COLOR,
                      self.line_thickness)
        
        return result
