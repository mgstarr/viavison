import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Waste Detection System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚀 Waste Detection System")
st.markdown("Real-time waste detection using YOLOv8 AI model with Robot Helper")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Higher = more confident detections (fewer false positives)"
    )
    
    use_webcam = st.checkbox("Use Webcam", value=True)
    
    if not use_webcam:
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    show_robot = st.checkbox("Show Robot Animation", value=True)

# Load model
@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()

waste_types = ['bottle', 'cup', 'can', 'box', 'trash', 'bag', 'plastic', 'paper', 'cardboard']

# Initialize session state
if 'waste_count' not in st.session_state:
    st.session_state.waste_count = {}
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0  # Only increases on successful collection
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = deque(maxlen=100)
if 'fps_history' not in st.session_state:
    st.session_state.fps_history = deque(maxlen=30)
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'last_waste_detection' not in st.session_state:
    st.session_state.last_waste_detection = None
if 'last_waste_name' not in st.session_state:
    st.session_state.last_waste_name = None
if 'animation_time' not in st.session_state:
    st.session_state.animation_time = 0
if 'is_reaching' not in st.session_state:
    st.session_state.is_reaching = False
if 'collected_items' not in st.session_state:
    st.session_state.collected_items = deque(maxlen=20)  # Recent collected items

def draw_robot_animation(frame, is_reaching, last_waste_position, animation_time, current_time):
    """Draw robot with animation reaching toward waste
    Returns: (frame, is_reaching, last_waste_position, animation_time, collection_success)
    """
    
    annotated_frame = frame.copy()
    robot_base = (80, annotated_frame.shape[0] - 80)
    shoulder = (robot_base[0] + 40, robot_base[1] - 60)
    collection_success = False
    
    # Draw robot arm animation
    if is_reaching and last_waste_position:
        elapsed = current_time - animation_time
        progress = min(elapsed / 1.2, 1.0)  # 1.2 second animation
        
        target = last_waste_position
        
        # Smooth easing function
        ease_progress = progress if progress < 0.5 else 1 - (1 - progress) ** 2
        
        # Elbow position
        elbow_x = shoulder[0] + (target[0] - shoulder[0]) * ease_progress * 0.5
        elbow_y = shoulder[1] + (target[1] - shoulder[1]) * ease_progress * 0.4
        elbow = (int(elbow_x), int(elbow_y))
        
        # Hand position
        hand_x = shoulder[0] + (target[0] - shoulder[0]) * ease_progress
        hand_y = shoulder[1] + (target[1] - shoulder[1]) * ease_progress
        hand = (int(hand_x), int(hand_y))
        
        # Draw robot body with active color
        cv2.rectangle(annotated_frame, 
                     (robot_base[0] - 25, robot_base[1] - 80),
                     (robot_base[0] + 25, robot_base[1]),
                     (50, 200, 255), -1)
        cv2.rectangle(annotated_frame, 
                     (robot_base[0] - 25, robot_base[1] - 80),
                     (robot_base[0] + 25, robot_base[1]),
                     (0, 255, 0), 3)
        
        # Draw robot head
        cv2.circle(annotated_frame, (robot_base[0], robot_base[1] - 95), 15, (50, 200, 255), -1)
        cv2.circle(annotated_frame, (robot_base[0], robot_base[1] - 95), 15, (0, 255, 0), 3)
        
        # Draw animated eyes (looking at target)
        eye_color = (0, 255, 0) if progress < 0.7 else (0, 255, 255)  # Green then cyan
        cv2.circle(annotated_frame, (robot_base[0] - 5, robot_base[1] - 97), 3, eye_color, -1)
        cv2.circle(annotated_frame, (robot_base[0] + 5, robot_base[1] - 97), 3, eye_color, -1)
        
        # Draw arm segments with gradient colors
        cv2.line(annotated_frame, shoulder, elbow, (0, 255, 150), 4)
        cv2.line(annotated_frame, elbow, hand, (0, 200, 255), 4)
        
        # Draw gripper (closes as it reaches)
        gripper_size = 10 - int(5 * ease_progress)  # Gets smaller as it reaches
        cv2.circle(annotated_frame, hand, max(gripper_size, 5), (0, 255, 100), -1)
        
        # Draw reaching line to waste
        cv2.line(annotated_frame, hand, target, (0, 255, 0), 2)
        cv2.circle(annotated_frame, target, 15, (0, 255, 0), 2)
        
        # Add progress bar with color change
        bar_width = 200
        bar_x = 10
        bar_y = 50
        filled_width = int(bar_width * ease_progress)
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (100, 100, 100), 1)
        
        # Progress bar color changes: yellow -> green at completion
        bar_color = (0, int(255 * ease_progress), 255 - int(255 * ease_progress))
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + 15), bar_color, -1)
        
        # Status text with percentage
        progress_pct = int(progress * 100)
        cv2.putText(annotated_frame, f"COLLECTING... {progress_pct}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Return flag if animation completed
        if progress >= 1.0:
            # Animation complete - collection successful!
            cv2.putText(annotated_frame, "WASTE COLLECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 3)
            collection_success = True
            return annotated_frame, False, None, 0, collection_success
        
        return annotated_frame, True, last_waste_position, animation_time, False
    
    else:
        # Draw idle robot
        cv2.rectangle(annotated_frame, 
                     (robot_base[0] - 25, robot_base[1] - 80),
                     (robot_base[0] + 25, robot_base[1]),
                     (100, 150, 200), -1)
        cv2.rectangle(annotated_frame, 
                     (robot_base[0] - 25, robot_base[1] - 80),
                     (robot_base[0] + 25, robot_base[1]),
                     (100, 100, 255), 2)
        
        # Draw robot head
        cv2.circle(annotated_frame, (robot_base[0], robot_base[1] - 95), 15, (100, 150, 200), -1)
        cv2.circle(annotated_frame, (robot_base[0], robot_base[1] - 95), 15, (100, 100, 255), 2)
        
        # Draw idle eyes (scanning)
        cv2.circle(annotated_frame, (robot_base[0] - 5, robot_base[1] - 97), 3, (0, 150, 200), -1)
        cv2.circle(annotated_frame, (robot_base[0] + 5, robot_base[1] - 97), 3, (0, 150, 200), -1)
        
        # Draw idle arm
        idle_elbow = (shoulder[0] + 30, shoulder[1] + 20)
        idle_hand = (shoulder[0] + 50, shoulder[1] + 30)
        cv2.line(annotated_frame, shoulder, idle_elbow, (0, 150, 200), 4)
        cv2.line(annotated_frame, idle_elbow, idle_hand, (0, 150, 200), 4)
        cv2.circle(annotated_frame, idle_hand, 8, (0, 150, 200), -1)
        
        # Add status text
        cv2.putText(annotated_frame, "SCANNING FOR WASTE...", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        return annotated_frame, False, None, 0, False

# Create columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Real-time Stats")
    stats_placeholder = st.empty()

# Create bottom row for additional info
bottom_col1, bottom_col2, bottom_col3 = st.columns(3)

with bottom_col1:
    st.subheader("🎯 Collection Status")
    collection_placeholder = st.empty()

with bottom_col2:
    st.subheader("📈 Top Waste Types")
    waste_types_placeholder = st.empty()

with bottom_col3:
    st.subheader("⏱️ Session Info")
    session_info_placeholder = st.empty()

# Main processing
if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    if uploaded_file is not None:
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file_path)
    else:
        st.warning("Please upload a video file")
        st.stop()

if not cap.isOpened():
    st.error("Cannot open webcam/video")
    st.stop()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start button
start_button = st.button("▶️ Start Detection", key="start_btn", use_container_width=True)
stop_button = st.button("⏹️ Stop Detection", key="stop_btn", use_container_width=True)

if start_button:
    st.session_state.detection_active = True

if stop_button:
    st.session_state.detection_active = False

if 'detection_active' not in st.session_state:
    st.session_state.detection_active = True

# Processing loop
frame_count = 0
last_time = time.time()

if st.session_state.detection_active:
    st.info("🟢 Detection is running... Press 'Stop Detection' to pause")
    
    while st.session_state.detection_active:
        ret, frame = cap.read()
        
        if not ret:
            if not use_webcam:
                st.success("Video playback completed!")
            break
        
        frame_count += 1
        current_time = time.time()
        fps = 1.0 / (current_time - last_time + 0.001)
        st.session_state.fps_history.append(fps)
        last_time = current_time
        
        # Run inference
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()
        
        # Check for waste
        waste_detected = False
        closest_waste = None
        
        for detection in results[0].boxes:
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]
            confidence = float(detection.conf[0])
            
            if any(waste_type in class_name.lower() for waste_type in waste_types):
                waste_detected = True
                
                x1, y1, x2, y2 = detection.xyxy[0]
                waste_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Add to detection history
                st.session_state.detection_history.append((waste_center, class_name, confidence, current_time))
                
                if closest_waste is None or confidence > closest_waste[2]:
                    closest_waste = (waste_center, class_name, confidence)
        
        # Update waste detection
        if waste_detected and closest_waste:
            st.session_state.last_waste_detection = closest_waste[0]
            st.session_state.last_waste_name = closest_waste[1]
            if not st.session_state.is_reaching:
                st.session_state.animation_time = current_time
                st.session_state.is_reaching = True
        
        st.session_state.frame_count = frame_count
        
        # Add robot animation if enabled
        if show_robot:
            annotated_frame, is_reaching, last_waste, anim_time, collection_success = draw_robot_animation(
                annotated_frame,
                st.session_state.is_reaching,
                st.session_state.last_waste_detection,
                st.session_state.animation_time,
                current_time
            )
            
            # Only increment counter when collection is successful!
            if collection_success:
                st.session_state.total_detections += 1
                
                # Add to waste type counter
                if st.session_state.last_waste_name:
                    if st.session_state.last_waste_name not in st.session_state.waste_count:
                        st.session_state.waste_count[st.session_state.last_waste_name] = 0
                    st.session_state.waste_count[st.session_state.last_waste_name] += 1
                    
                    # Add to collected items list
                    st.session_state.collected_items.append({
                        'name': st.session_state.last_waste_name,
                        'time': datetime.now().strftime("%H:%M:%S")
                    })
            
            # Update session state
            st.session_state.is_reaching = is_reaching
            if not is_reaching:
                st.session_state.last_waste_detection = None
                st.session_state.last_waste_name = None
        
        # Add text overlays to frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if closest_waste:
            cv2.putText(annotated_frame, f"Detected: {closest_waste[1]} ({closest_waste[2]:.2f})", 
                       (250, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 0), 2)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Update video placeholder
        video_placeholder.image(frame_rgb, use_column_width=True)
        
        # Update stats
        with stats_placeholder.container():
            st.metric("✅ Collected", st.session_state.total_detections)
            if len(st.session_state.fps_history) > 0:
                avg_fps = np.mean(st.session_state.fps_history)
                st.metric("Avg FPS", f"{avg_fps:.1f}")
            if closest_waste:
                st.metric("Latest Detection", closest_waste[1])
                st.metric("Confidence", f"{closest_waste[2]:.2%}")
        
        # Update collection status
        with collection_placeholder.container():
            if st.session_state.is_reaching:
                st.warning("🤖 Robot is collecting waste...")
            else:
                if st.session_state.total_detections > 0:
                    st.success(f"✅ {st.session_state.total_detections} items collected")
                else:
                    st.info("Waiting for waste detection...")
            
            # Show recent collections
            if st.session_state.collected_items:
                st.write("**Recent Collections:**")
                for item in list(st.session_state.collected_items)[-3:]:
                    st.write(f"• {item['name']} ({item['time']})")
        
        # Update waste types (by collection)
        sorted_waste = sorted(st.session_state.waste_count.items(), key=lambda x: x[1], reverse=True)[:5]
        with waste_types_placeholder.container():
            if sorted_waste:
                for waste_type, count in sorted_waste:
                    st.write(f"**{waste_type}**: {count}")
            else:
                st.write("No waste collected yet")
        
        # Update session info
        elapsed = current_time - st.session_state.start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        with session_info_placeholder.container():
            st.write(f"⏱️ **Duration**: {minutes:02d}:{seconds:02d}")
            st.write(f"📊 **Frames**: {st.session_state.frame_count}")
            collection_rate = st.session_state.total_detections / max(elapsed, 1)
            st.write(f"🎯 **Rate**: {collection_rate:.2f}/sec")
        
        # Small delay
        time.sleep(0.01)

else:
    st.info("🔴 Detection is paused. Press 'Start Detection' to continue")

cap.release()

# Final report section

# Final report section
st.markdown("---")
st.subheader("📋 Session Report")

report_col1, report_col2 = st.columns(2)

with report_col1:
    st.metric("✅ Successfully Collected", st.session_state.total_detections)

with report_col2:
    if len(st.session_state.fps_history) > 0:
        st.metric("Average FPS", f"{np.mean(st.session_state.fps_history):.1f}")

# Waste breakdown table
if st.session_state.waste_count:
    st.subheader("Waste Collection Breakdown")
    
    waste_data = []
    total = sum(st.session_state.waste_count.values())
    for waste_type, count in sorted(st.session_state.waste_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        waste_data.append({
            "Waste Type": waste_type,
            "Collected": count,
            "Percentage": f"{percentage:.1f}%"
        })
    
    st.dataframe(waste_data, use_column_width=True)

# Download report button
if st.session_state.total_detections > 0:
    report_text = f"""
WASTE DETECTION SESSION REPORT
{'='*60}

Session Statistics:
- Successfully Collected: {st.session_state.total_detections}
- Total Frames Processed: {st.session_state.frame_count}
- Average FPS: {np.mean(st.session_state.fps_history):.1f}

Waste Collection Breakdown:
"""
    
    for waste_type, count in sorted(st.session_state.waste_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sum(st.session_state.waste_count.values()) * 100)
        report_text += f"\n{waste_type}: {count} ({percentage:.1f}%)"
    
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=f"waste_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Waste breakdown table
if st.session_state.waste_count:
    st.subheader("Waste Collection Breakdown")
    
    waste_data = []
    total = sum(st.session_state.waste_count.values())
    for waste_type, count in sorted(st.session_state.waste_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        waste_data.append({
            "Waste Type": waste_type,
            "Collected": count,
            "Percentage": f"{percentage:.1f}%"
        })
    
    st.dataframe(waste_data, use_column_width=True)

# Download report button
if st.session_state.total_detections > 0:
    report_text = f"""
WASTE DETECTION SESSION REPORT
{'='*60}

Session Statistics:
- Successfully Collected: {st.session_state.total_detections}
- Total Detected: {len(st.session_state.detection_history)}
- Total Frames Processed: {st.session_state.frame_count}
- Average FPS: {np.mean(st.session_state.fps_history):.1f}

Waste Collection Breakdown:
"""
    
    for waste_type, count in sorted(st.session_state.waste_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sum(st.session_state.waste_count.values()) * 100)
        report_text += f"\n{waste_type}: {count} ({percentage:.1f}%)"
    
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=f"waste_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )