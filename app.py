"""
Tactical Eye - Football Match Analysis (Streamlit)
Main entry point for the web application
Handles headless/server environments
"""

# IMPORTANT: Set headless mode BEFORE any other imports
import os
import sys

# Disable GUI backends
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # For headless servers
os.environ['MPLBACKEND'] = 'Agg'  # Matplotlib headless

import cv2
import yaml
import logging
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST
import streamlit as st

st.set_page_config(
    page_title="Tactical Eye - Football Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
        font-size: 2.5em;
    }
    
    h2 {
        color: #ff7f0e;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #2ca02c;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/highlights").mkdir(exist_ok=True)
Path("outputs/logs").mkdir(exist_ok=True)

# Initialize session state
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'detector' not in st.session_state:
    st.session_state.detector = None

# Main page
st.markdown("<h1>⚽ Tactical Eye - Football Match Analysis</h1>", unsafe_allow_html=True)
st.markdown("*Professional AI-powered tactical analysis system for football*")

# Sidebar info
st.sidebar.markdown("## About Tactical Eye")
st.sidebar.info(
    """
    **Tactical Eye** uses advanced computer vision (YOLOv8) to analyze football matches:
    
    - 🎯 Detect players and ball
    - 🎨 Classify teams by jersey color
    - 🔄 Track movement patterns
    - ⚡ Identify tactical events
    - 💾 Export highlights
    """
)

st.sidebar.markdown("---")

# Quick start
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🚀 Quick Start
    
    1. **Upload** your match video (MP4, AVI, MOV)
    2. **Configure** detection and perspective settings
    3. **Analyze** the video for tactical patterns
    4. **View** frame-by-frame detections
    5. **Export** highlight clips
    """)

with col2:
    st.markdown("""
    ### 📋 Supported Formats
    
    - **Video:** MP4, AVI, MOV, MKV
    - **Max Size:** 500 MB
    - **Min Duration:** 5 seconds
    - **Recommended:** HD (1920x1080)
    
    ### ⚙️ Requirements
    
    - Python 3.9+
    - CUDA (optional, for GPU)
    - 4GB RAM minimum
    """)

st.markdown("---")

# Features overview
st.markdown("### 🎯 Key Features")

feature_cols = st.columns(5)

features = [
    ("🎥", "Detection", "YOLOv8 object detection"),
    ("🎨", "Classification", "Team color analysis"),
    ("🔄", "Tracking", "Player movement tracking"),
    ("📊", "Analytics", "Tactical event detection"),
    ("💾", "Export", "Highlight clip generation")
]

for i, (emoji, title, desc) in enumerate(features):
    with feature_cols[i]:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
            <h3>{emoji}</h3>
            <strong>{title}</strong>
            <p style="font-size: 0.9em; color: #666;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Navigation info
st.markdown("""
### 📑 Navigation

Use the sidebar to navigate between pages:

- **📹 Upload Video** - Upload and preview your match footage
- **🎬 Frame Preview** - View frame-by-frame detections
- **📊 Analysis Results** - Detailed analysis statistics
- **⚡ Event Detection** - Tactical event timeline
- **💾 Export Highlights** - Generate and download clips
""")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #999; padding: 20px; border-top: 1px solid #ddd; margin-top: 30px;">
    <p>🚀 <strong>Get Started:</strong> Navigate to <strong>📹 Upload Video</strong> to begin analyzing</p>
    <p style="font-size: 0.9em;">© 2026 Tactical Eye | Advanced Football Analytics</p>
</div>
""", unsafe_allow_html=True)
