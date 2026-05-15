"""
Tactical Eye - Football Match Highlight Generator
A professional system for detecting, tracking, and analyzing tactical patterns in soccer
"""

__version__ = "1.0.0"
__author__ = "Tactical Eye Team"

from .detector import TacticalDetector
from .perspective import PerspectiveTransformer
from .tactical_overlay import TacticalOverlay
from .event_detection import EventDetector
from .video_export import VideoExporter

__all__ = [
    'TacticalDetector',
    'PerspectiveTransformer',
    'TacticalOverlay',
    'EventDetector',
    'VideoExporter'
]