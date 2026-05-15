"""
Step 6: Video Export with Effects
Exports event highlights with tactical overlays and slow-motion effects
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoExporter:
    """
    Exports video clips with tactical overlays and effects.
    Supports slow-motion, multi-clip compilation, and effect layering.
    """
    
    def __init__(self, output_dir: str = "outputs/highlights", fps: int = 30,
                 codec: str = "mp4v", output_format: str = "mp4"):
        """
        Initialize video exporter.
        
        Args:
            output_dir: Directory for output videos
            fps: Output video FPS
            codec: Video codec (mp4v, MJPG, etc.)
            output_format: Output format (mp4, avi, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.codec = codec
        self.output_format = output_format
        
        logger.info(f"VideoExporter initialized: {output_dir}")
    
    def export_event_clip(self, video_path: str, start_frame: int, end_frame: int,
                         output_name: str, apply_slowmo: bool = False,
                         slowmo_factor: float = 0.5) -> str:
        """
        Export a single event clip from the source video.
        
        Args:
            video_path: Path to source video
            start_frame: Start frame index
            end_frame: End frame index
            output_name: Name for output file (without extension)
            apply_slowmo: Whether to apply slow-motion effect
            slowmo_factor: Slow-motion factor (0.5 = 2x slower)
        
        Returns:
            Path to exported video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Prepare output
        output_path = self.output_dir / f"{output_name}.{self.output_format}"
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        
        # Adjust FPS for slow-motion
        output_fps = self.fps if not apply_slowmo else int(self.fps * slowmo_factor)
        
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps,
                            (frame_width, frame_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Cannot initialize video writer for {output_path}")
        
        # Extract and write frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        
        logger.info(f"Exporting clip: {output_name} ({end_frame - start_frame} frames)")
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            if apply_slowmo:
                # Write frame multiple times for slow-motion
                num_repeats = int(1 / slowmo_factor)
                for _ in range(num_repeats):
                    out.write(frame)
            else:
                out.write(frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Clip exported: {output_path}")
        return str(output_path)
    
    def export_with_overlays(self, video_path: str, events: List[Dict],
                            overlay_func, output_name: str) -> str:
        """
        Export video with tactical overlays applied to specified events.
        
        Args:
            video_path: Path to source video
            events: List of event dictionaries with start_frame and end_frame
            overlay_func: Function to apply overlays (takes frame, detections)
            output_name: Output file name (without extension)
        
        Returns:
            Path to exported video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = self.output_dir / f"{output_name}.{self.output_format}"
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                            (frame_width, frame_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Cannot initialize video writer for {output_path}")
        
        # Create event frame set for quick lookup
        event_frames = set()
        for event in events:
            for f in range(event['start_frame'], event['end_frame']):
                event_frames.add(f)
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Exporting with overlays: {output_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply overlays only to event frames
            if frame_count in event_frames:
                frame = overlay_func(frame)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Overlay video exported: {output_path}")
        return str(output_path)
    
    def export_slow_motion_segment(self, video_path: str, start_frame: int,
                                   end_frame: int, output_name: str,
                                   slowmo_factor: float = 0.5) -> str:
        """
        Export a segment with slow-motion effect.
        
        Args:
            video_path: Path to source video
            start_frame: Start frame
            end_frame: End frame
            output_name: Output file name
            slowmo_factor: Slow-motion factor (0.5 = 2x slower)
        
        Returns:
            Path to exported video
        """
        return self.export_event_clip(
            video_path, start_frame, end_frame, output_name,
            apply_slowmo=True, slowmo_factor=slowmo_factor
        )
    
    def compile_highlight_reel(self, clips: List[str], output_name: str) -> str:
        """
        Compile multiple video clips into a single highlight reel.
        
        Args:
            clips: List of clip file paths
            output_name: Output file name
        
        Returns:
            Path to compiled video
        """
        if not clips:
            raise ValueError("No clips provided for compilation")
        
        output_path = self.output_dir / f"{output_name}.{self.output_format}"
        
        logger.info(f"Compiling highlight reel from {len(clips)} clips")
        
        # For simple compilation, use OpenCV frame concatenation
        all_frames = []
        frame_properties = None
        
        for clip_path in clips:
            cap = cv2.VideoCapture(clip_path)
            
            if not cap.isOpened():
                logger.warning(f"Cannot open clip: {clip_path}")
                continue
            
            if frame_properties is None:
                frame_properties = {
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            
            cap.release()
        
        # Write compiled video
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                            (frame_properties['width'], frame_properties['height']))
        
        for frame in all_frames:
            out.write(frame)
        
        out.release()
        
        logger.info(f"Highlight reel compiled: {output_path} ({len(all_frames)} frames)")
        return str(output_path)