"""
Main execution script for Tactical Eye highlight generator
Orchestrates detection, tracking, analysis, and export pipeline
"""
import cv2
import yaml
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# 파일들이 모두 같은 위치(root)에 있으므로 직접 import 합니다.
try:
    from detector import TacticalDetector
    from perspective import PerspectiveTransformer
    from tactical_overlay import TacticalOverlay
    from event_detection import EventDetector
    from video_export import VideoExporter
    # 만약 다른 파일에서 utils를 참조한다면 해당 파일들도 수정이 필요할 수 있습니다.
except ImportError as e:
    logging.error(f"Import failed: {e}")
    # 현재 경로를 시스템 경로에 강제로 추가 (보험용)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from detector import TacticalDetector
    from perspective import PerspectiveTransformer
    from tactical_overlay import TacticalOverlay
    from event_detection import EventDetector
    from video_export import VideoExporter




import cv2
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from src.detector import TacticalDetector
from src.perspective import PerspectiveTransformer
from src.tactical_overlay import TacticalOverlay
from src.event_detection import EventDetector
from src.video_export import VideoExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/tactical_eye.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TacticalEyeAnalyzer:
    """
    Main coordinator for football tactical analysis pipeline.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize analyzer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Configuration loaded from " + config_path)
        
        # Initialize components
        self.detector = TacticalDetector(
            model_name=self.config['detection']['model'],
            conf_threshold=self.config['detection']['conf_threshold'],
            device=self.config['detection']['device'],
            kmeans_clusters=self.config['team_classification']['kmeans_clusters']
        )
        
        self.perspective = PerspectiveTransformer(
            pitch_length=self.config['perspective']['pitch_length'],
            pitch_width=self.config['perspective']['pitch_width'],
            output_width=self.config['perspective']['output_width'],
            output_height=self.config['perspective']['output_height']
        )
        
        self.overlay = TacticalOverlay(
            line_thickness=self.config['tactical']['line_thickness'],
            centroid_radius=self.config['tactical']['centroid_radius'],
            hull_alpha=self.config['tactical']['hull_alpha'],
            text_scale=self.config['tactical']['text_scale']
        )
        
        self.event_detector = EventDetector(
            high_pressing_threshold=self.config['event_detection']['high_pressing_threshold'],
            min_players_pressing=self.config['event_detection']['min_players_pressing'],
            transition_time_window=self.config['event_detection']['transition_time_window'],
            min_event_duration=self.config['event_detection']['min_event_duration']
        )
        
        self.exporter = VideoExporter(
            fps=self.config['video_export']['output_fps'],
            codec=self.config['video_export']['codec']
        )
        
        logger.info("TacticalEyeAnalyzer initialized successfully")
    
    def analyze_video(self, video_path: str, source_points: List[tuple] = None) -> Dict:
        """
        Analyze a complete football match video.
        
        Args:
            video_path: Path to input video
            source_points: Perspective transform source points (pitch corners)
        
        Returns:
            Analysis results dictionary
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Analyzing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Set perspective source points if provided
        if source_points:
            self.perspective.set_source_points(source_points)
        
        # Analysis results
        results = {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'events': [],
            'detections_history': [],
            'tactical_maps': []
        }
        
        # Create tactical map template
        tactical_map_template = self.perspective.create_empty_tactical_map()
        
        # Process video
        frame_idx = 0
        with tqdm(total=total_frames, desc="Analyzing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detection
                detections = self.detector.detect_frame(frame)
                detections = self.detector.classify_teams(frame, detections,
                                                         self.config['team_classification']['color_space'])
                team_stats = self.detector.get_team_statistics(detections)
                
                # Store detections
                results['detections_history'].append(detections)
                
                # Event detection
                event_report = self.event_detector.generate_event_report(
                    frame_idx, fps, detections, results['detections_history']
                )
                
                if event_report['events']:
                    results['events'].append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'event_data': event_report
                    })
                
                # Tactical overlays
                annotated = self.overlay.draw_detections(frame, detections)
                annotated = self.overlay.draw_team_shapes(annotated, detections)
                annotated = self.overlay.draw_centroids(annotated, team_stats)
                annotated = self.overlay.draw_ball_to_nearest_player(annotated, detections)
                
                # Tactical map
                if source_points:
                    tactical_map = self.overlay.draw_on_tactical_map(
                        tactical_map_template.copy(), detections, self.perspective
                    )
                    results['tactical_maps'].append(tactical_map)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        logger.info(f"Analysis complete. Detected {len(results['events'])} events.")
        return results
    
    def export_highlights(self, results: Dict, output_prefix: str = "highlight") -> List[str]:
        """
        Export detected events as individual video clips.
        
        Args:
            results: Analysis results from analyze_video()
            output_prefix: Prefix for output files
        
        Returns:
            List of exported video paths
        """
        exported_clips = []
        
        video_path = results['video_path']
        fps = results['fps']
        
        logger.info(f"Exporting {len(results['events'])} highlight clips")
        
        for i, event in enumerate(results['events']):
            frame_idx = event['frame_idx']
            timestamp = event['timestamp']
            
            # Define clip boundaries (2 seconds before and after event)
            buffer_frames = int(fps * 2)
            start_frame = max(0, frame_idx - buffer_frames)
            end_frame = min(results['total_frames'], frame_idx + buffer_frames)
            
            output_name = f"{output_prefix}_{i:03d}_t{timestamp:.2f}s"
            
            # Apply slow-motion around event
            clip_path = self.exporter.export_event_clip(
                video_path, start_frame, end_frame, output_name,
                apply_slowmo=True, slowmo_factor=self.config['video_export']['slowmo_factor']
            )
            
            exported_clips.append(clip_path)
        
        logger.info(f"Exported {len(exported_clips)} highlight clips")
        return exported_clips
    
    def export_highlight_reel(self, clips: List[str], output_name: str = "tactical_highlights") -> str:
        """
        Compile multiple highlight clips into a single reel.
        
        Args:
            clips: List of clip paths
            output_name: Output reel name
        
        Returns:
            Path to compiled reel
        """
        return self.exporter.compile_highlight_reel(clips, output_name)


def main():
    """Main execution"""
    
    # Configuration
    VIDEO_PATH = "match_footage.mp4"  # Replace with your video
    SOURCE_POINTS = [
        (120, 240),   # Top-left corner of pitch
        (1800, 200),  # Top-right corner
        (1900, 1050), # Bottom-right corner
        (50, 1000)    # Bottom-left corner
    ]
    
    # Initialize analyzer
    analyzer = TacticalEyeAnalyzer(config_path="config.yaml")
    
    # Analyze video
    logger.info("=" * 60)
    logger.info("TACTICAL EYE HIGHLIGHT GENERATOR")
    logger.info("=" * 60)
    
    results = analyzer.analyze_video(VIDEO_PATH, source_points=SOURCE_POINTS)
    
    # Export highlights
    clips = analyzer.export_highlights(results)
    
    # Compile highlight reel
    if clips:
        reel_path = analyzer.export_highlight_reel(clips)
        logger.info(f"✓ Highlight reel created: {reel_path}")
    
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
