"""
Upload Video Page
Handles video file upload, preview, and configuration
"""

# Headless setup
import os
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import streamlit as st
import cv2
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Upload Video", page_icon="📹")

st.markdown("# 📹 Upload Match Video")
st.markdown("Upload a football match video for analysis")

# Create upload directory
Path("uploads").mkdir(exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(
    "Select a video file",
    type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
    help="Recommended: HD resolution (1920x1080), H.264 codec"
)

if uploaded_file:
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    
    st.session_state.video_file = {
        'path': video_path,
        'name': uploaded_file.name,
        'size': uploaded_file.size
    }
    
    try:
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Failed to open video file. Try a different format.")
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Success message
            st.success(f"✓ Video uploaded: **{uploaded_file.name}**")
            
            # Video preview
            st.markdown("### Video Preview")
            st.video(video_path)
            
            # Video properties
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Resolution", f"{width}×{height}")
            with col2:
                st.metric("FPS", fps)
            with col3:
                st.metric("Total Frames", total_frames)
            with col4:
                st.metric("Duration", f"{duration:.1f}s")
            with col5:
                st.metric("File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
            
            cap.release()
            
            st.markdown("---")
            
            # Analysis settings
            st.markdown("### ⚙️ Analysis Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Processing")
                max_frames = st.number_input(
                    "Max frames to process",
                    min_value=10,
                    max_value=min(10000, total_frames),
                    value=min(300, total_frames),
                    help="Process first N frames (reduce for faster testing)"
                )
                
                sample_frames = st.slider(
                    "Sample every N frames",
                    min_value=1,
                    max_value=30,
                    value=1,
                    help="1 = every frame, 5 = every 5th frame (faster)"
                )
            
            with col2:
                st.markdown("#### Detection")
                model_size = st.selectbox(
                    "YOLOv8 Model Size",
                    ["yolov8n.pt (fastest)", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt (best)"],
                    index=0,
                    help="Larger models are more accurate but slower"
                )
                
                conf_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum confidence for detections"
                )
            
            st.markdown("---")
            
            # Perspective transform setup
            st.markdown("### 🎯 Perspective Transform (Optional)")
            
            use_perspective = st.checkbox(
                "Enable bird's-eye view conversion",
                value=True,
                help="Convert field view to top-down tactical map"
            )
            
            if use_perspective:
                st.info("""
                🔍 **How to set perspective points:**
                
                1. Identify the 4 corners of the football pitch in your video
                2. Enter the pixel coordinates (x, y) for each corner
                3. Standard pitch: 105m × 68m
                
                **Format:** Enter as `x,y` (e.g., `120,240`)
                """)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Top-Left**")
                    tl_x = st.number_input("TL-X", value=120, key="tl_x")
                    tl_y = st.number_input("TL-Y", value=240, key="tl_y")
                
                with col2:
                    st.markdown("**Top-Right**")
                    tr_x = st.number_input("TR-X", value=1800, key="tr_x")
                    tr_y = st.number_input("TR-Y", value=200, key="tr_y")
                
                with col3:
                    st.markdown("**Bottom-Right**")
                    br_x = st.number_input("BR-X", value=1900, key="br_x")
                    br_y = st.number_input("BR-Y", value=1050, key="br_y")
                
                with col4:
                    st.markdown("**Bottom-Left**")
                    bl_x = st.number_input("BL-X", value=50, key="bl_x")
                    bl_y = st.number_input("BL-Y", value=1000, key="bl_y")
                
                # Store perspective points in session state
                st.session_state.perspective_points = [
                    (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)
                ]
            else:
                st.session_state.perspective_points = None
            
            st.markdown("---")
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Perspective Transform")
                    pitch_length = st.number_input("Pitch Length (m)", min_value=100, max_value=120, value=105)
                    pitch_width = st.number_input("Pitch Width (m)", min_value=60, max_value=80, value=68)
                
                with col2:
                    st.markdown("#### Event Detection")
                    high_pressing_threshold = st.slider(
                        "High-Pressing Threshold",
                        0.0, 1.0, 0.5, 0.05,
                        help="Distance threshold for detecting pressing"
                    )
                    min_players_pressing = st.slider(
                        "Min Players for Pressing",
                        1, 11, 3,
                        help="Minimum players near ball"
                    )
                
                st.session_state.advanced_settings = {
                    'pitch_length': pitch_length,
                    'pitch_width': pitch_width,
                    'high_pressing_threshold': high_pressing_threshold,
                    'min_players_pressing': min_players_pressing
                }
            
            st.markdown("---")
            
            # Start analysis button
            if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
                # Store settings in session state
                model_name = model_size.split('(')[0].strip()
                
                st.session_state.analysis_settings = {
                    'max_frames': max_frames,
                    'sample_frames': sample_frames,
                    'model_name': model_name,
                    'conf_threshold': conf_threshold,
                    'use_perspective': use_perspective,
                    'perspective_points': st.session_state.perspective_points if use_perspective else None
                }
                
                st.success("✓ Analysis started! Navigate to **🎬 Frame Preview** to view results.")
                st.info("This may take a few minutes depending on video length...")

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        logger.error(f"Video processing error: {e}")

else:
    st.info("👆 Upload a video file to get started")