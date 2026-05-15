"""
Step 2: Detection & Team Classification
Detects players, referees, and ball using YOLOv8
Classifies players into Home Team and Away Team using K-Means clustering on jersey colors
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TacticalDetector:
    """
    YOLOv8-based detector for football match analysis.
    Detects players, referees, and ball, then classifies teams by jersey color.
    """
    
    # COCO class indices
    PERSON_CLASS = 0
    SPORTS_BALL_CLASS = 32
    
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5, 
                 device: str = "0", kmeans_clusters: int = 2):
        """
        Initialize the tactical detector.
        
        Args:
            model_name: YOLOv8 model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            conf_threshold: Confidence threshold for detections
            device: GPU device ID or "cpu"
            kmeans_clusters: Number of K-Means clusters for team classification (typically 2)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self.kmeans_clusters = kmeans_clusters
        
        logger.info(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        
        logger.info("Tactical detector initialized")
    
    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Run YOLOv8 detection on a single frame.
        
        Args:
            frame: Input frame (BGR, np.ndarray)
            
        Returns:
            Dictionary containing detection results
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = {
            'players': [],
            'ball': None,
            'referees': []
        }
        
        # Extract detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                bbox = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'class_id': class_id,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                
                if class_id == self.PERSON_CLASS:
                    # Extract player crop for color analysis
                    player_crop = frame[y1:y2, x1:x2]
                    bbox['crop'] = player_crop
                    detections['players'].append(bbox)
                    
                elif class_id == self.SPORTS_BALL_CLASS:
                    detections['ball'] = bbox
        
        logger.debug(f"Detected {len(detections['players'])} players, "
                    f"ball: {detections['ball'] is not None}")
        
        return detections
    
    def classify_teams(self, frame: np.ndarray, detections: Dict, 
                      color_space: str = "hsv") -> Dict:
        """
        Classify detected players into Home Team and Away Team using K-Means clustering
        on dominant jersey colors.
        
        Args:
            frame: Original frame (BGR)
            detections: Detection results from detect_frame()
            color_space: Color space for analysis ("hsv", "lab", "rgb")
            
        Returns:
            Updated detections with team classification
        """
        if not detections['players']:
            logger.warning("No players detected for team classification")
            return detections
        
        players = detections['players']
        color_features = []
        
        # Extract color features from player crops
        for player in players:
            crop = player['crop']
            
            if color_space == "hsv":
                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                # Calculate mean color in HSV space
                mean_color = cv2.mean(crop_hsv)[:3]
                
            elif color_space == "lab":
                crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                mean_color = cv2.mean(crop_lab)[:3]
                
            else:  # RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mean_color = cv2.mean(crop_rgb)[:3]
            
            color_features.append(mean_color)
        
        color_features = np.array(color_features)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=min(self.kmeans_clusters, len(players)), 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(color_features)
        
        # Assign team labels (0 = Home Team, 1 = Away Team)
        for i, player in enumerate(players):
            player['team'] = int(labels[i])
            player['team_name'] = 'Home Team' if labels[i] == 0 else 'Away Team'
            
            # Calculate dominant color
            if color_space == "hsv":
                dominant_h = int(color_features[i][0])
                dominant_color_name = self._hue_to_color_name(dominant_h)
            else:
                dominant_color_name = "Unknown"
            
            player['dominant_color'] = dominant_color_name
        
        logger.info(f"Team classification complete: "
                   f"{sum(1 for p in players if p['team'] == 0)} vs "
                   f"{sum(1 for p in players if p['team'] == 1)}")
        
        return detections
    
    @staticmethod
    def _hue_to_color_name(hue: int) -> str:
        """Convert HSV hue value to color name."""
        if 0 <= hue < 15 or hue >= 345:
            return "Red"
        elif 15 <= hue < 45:
            return "Orange"
        elif 45 <= hue < 75:
            return "Yellow"
        elif 75 <= hue < 165:
            return "Green"
        elif 165 <= hue < 255:
            return "Blue"
        elif 255 <= hue < 345:
            return "Purple"
        return "Unknown"
    
    def get_team_statistics(self, detections: Dict) -> Dict:
        """
        Calculate statistics for each team.
        
        Args:
            detections: Detection results with team classification
            
        Returns:
            Dictionary with team statistics
        """
        stats = {
            'home_team': {
                'count': 0,
                'positions': [],
                'centroid': None
            },
            'away_team': {
                'count': 0,
                'positions': [],
                'centroid': None
            }
        }
        
        for player in detections['players']:
            team_key = 'home_team' if player['team'] == 0 else 'away_team'
            stats[team_key]['count'] += 1
            stats[team_key]['positions'].append([player['center_x'], player['center_y']])
        
        # Calculate centroids
        for team_key in ['home_team', 'away_team']:
            if stats[team_key]['positions']:
                positions = np.array(stats[team_key]['positions'])
                stats[team_key]['centroid'] = tuple(positions.mean(axis=0).astype(int))
        
        logger.debug(f"Team stats: Home={stats['home_team']['count']}, "
                    f"Away={stats['away_team']['count']}")
        
        return stats