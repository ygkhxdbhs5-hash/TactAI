"""
Step 5: Event Detection Logic
Identifies tactical highlights: high-pressing, fast transitions, shots
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EventDetector:
    """
    Detects tactical events from player and ball tracking data.
    Identifies: high-pressing, transitions, shots, pass sequences
    """
    
    def __init__(self, high_pressing_threshold: float = 0.5,
                 min_players_pressing: int = 3,
                 transition_time_window: float = 2.0,
                 min_event_duration: float = 1.0):
        """
        Initialize event detector.
        
        Args:
            high_pressing_threshold: Distance threshold (0-1, normalized) for pressing
            min_players_pressing: Minimum players needed near ball for high-pressing
            transition_time_window: Time window (seconds) to detect transitions
            min_event_duration: Minimum event duration in seconds
        """
        self.high_pressing_threshold = high_pressing_threshold
        self.min_players_pressing = min_players_pressing
        self.transition_time_window = transition_time_window
        self.min_event_duration = min_event_duration
        
        self.event_history = []
        logger.info("Event detector initialized")
    
    def detect_high_pressing(self, detections: Dict, frame_width: int, 
                            frame_height: int) -> bool:
        """
        Detect high-pressing moment (multiple defenders near ball carrier).
        
        Args:
            detections: Detection results with team classification
            frame_width: Video frame width
            frame_height: Video frame height
        
        Returns:
            True if high-pressing detected
        """
        if not detections['ball'] or not detections['players']:
            return False
        
        ball = detections['ball']
        ball_pos = np.array([ball['center_x'], ball['center_y']])
        
        # Calculate normalized frame diagonal (max possible distance)
        max_distance = np.sqrt(frame_width**2 + frame_height**2)
        threshold_pixels = self.high_pressing_threshold * max_distance
        
        # Count players near ball
        pressing_players = 0
        for player in detections['players']:
            player_pos = np.array([player['center_x'], player['center_y']])
            distance = np.linalg.norm(ball_pos - player_pos)
            
            if distance < threshold_pixels:
                pressing_players += 1
        
        is_pressing = pressing_players >= self.min_players_pressing
        
        if is_pressing:
            logger.debug(f"High-pressing detected: {pressing_players} players near ball")
        
        return is_pressing
    
    def detect_transition(self, detections_history: List[Dict], fps: int) -> Optional[Dict]:
        """
        Detect fast transition (quick change of ball possession/direction).
        
        Args:
            detections_history: List of detection results over time
            fps: Video frames per second
        
        Returns:
            Event dictionary if transition detected, None otherwise
        """
        if len(detections_history) < fps * self.transition_time_window:
            return None
        
        # Get recent ball positions
        window_frames = int(fps * self.transition_time_window)
        recent_history = detections_history[-window_frames:]
        
        ball_positions = []
        for detections in recent_history:
            if detections['ball']:
                ball = detections['ball']
                ball_positions.append([ball['center_x'], ball['center_y']])
        
        if len(ball_positions) < 2:
            return None
        
        ball_positions = np.array(ball_positions)
        
        # Calculate ball movement variance (fast transitions have erratic movement)
        movement_variance = np.var(np.diff(ball_positions, axis=0), axis=0).mean()
        
        # Heuristic: significant variance indicates transition
        if movement_variance > 50:  # Tunable threshold
            logger.debug(f"Transition detected: movement variance = {movement_variance:.2f}")
            return {
                'event_type': 'transition',
                'confidence': min(movement_variance / 100, 1.0),
                'description': 'Fast transition detected'
            }
        
        return None
    
    def detect_shot_attempt(self, detections: Dict, frame_width: int,
                           frame_height: int) -> bool:
        """
        Detect potential shot attempt (ball near goal line with players nearby).
        
        Args:
            detections: Detection results
            frame_width: Video frame width
            frame_height: Video frame height
        
        Returns:
            True if shot attempt likely
        """
        if not detections['ball'] or not detections['players']:
            return False
        
        ball = detections['ball']
        
        # Check if ball is near goal line
        goal_line_distance = min(ball['center_x'], frame_width - ball['center_x'])
        near_goal = goal_line_distance < frame_width * 0.15
        
        if not near_goal:
            return False
        
        # Check if players are in shooting position
        ball_pos = np.array([ball['center_x'], ball['center_y']])
        
        for player in detections['players']:
            player_pos = np.array([player['center_x'], player['center_y']])
            distance = np.linalg.norm(ball_pos - player_pos)
            
            if distance < frame_width * 0.08:  # Close to ball
                logger.debug("Shot attempt detected")
                return True
        
        return False
    
    def analyze_defensive_line(self, detections: Dict) -> Dict:
        """
        Analyze defensive line position and compactness.
        
        Args:
            detections: Detection results
        
        Returns:
            Dictionary with defensive line metrics
        """
        players_by_team = {0: [], 1: []}
        
        for player in detections['players']:
            players_by_team[player['team']].append([player['center_x'], player['center_y']])
        
        metrics = {
            'home_team': self._calculate_line_metrics(players_by_team[0]),
            'away_team': self._calculate_line_metrics(players_by_team[1])
        }
        
        return metrics
    
    @staticmethod
    def _calculate_line_metrics(player_positions: List[List[int]]) -> Dict:
        """Calculate metrics for a team's defensive line."""
        if not player_positions:
            return {'count': 0, 'avg_x': 0, 'spacing': 0, 'compactness': 0}
        
        positions = np.array(player_positions)
        
        avg_x = positions[:, 0].mean()
        avg_y = positions[:, 1].mean()
        
        # Spacing: average distance between players
        if len(positions) > 1:
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            spacing = np.mean(distances)
        else:
            spacing = 0
        
        # Compactness: std dev of y-positions (how spread out vertically)
        compactness = positions[:, 1].std()
        
        return {
            'count': len(player_positions),
            'avg_x': avg_x,
            'avg_y': avg_y,
            'spacing': spacing,
            'compactness': compactness
        }
    
    def generate_event_report(self, frame_idx: int, fps: int, detections: Dict,
                             detections_history: List[Dict]) -> Dict:
        """
        Generate comprehensive event report for current frame.
        
        Args:
            frame_idx: Current frame index
            fps: Video FPS
            detections: Current frame detections
            detections_history: Historical detections
        
        Returns:
            Event report dictionary
        """
        timestamp = frame_idx / fps
        
        report = {
            'timestamp': timestamp,
            'frame_idx': frame_idx,
            'events': [],
            'metrics': {}
        }
        
        # Check for high-pressing
        if self.detect_high_pressing(detections, 1920, 1080):  # Default HD resolution
            report['events'].append('high_pressing')
        
        # Check for transition
        transition = self.detect_transition(detections_history, fps)
        if transition:
            report['events'].append(transition)
        
        # Check for shot
        if self.detect_shot_attempt(detections, 1920, 1080):
            report['events'].append('shot_attempt')
        
        # Analyze defensive lines
        report['metrics'] = self.analyze_defensive_line(detections)
        
        return report