import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# --- Page Config ---
st.set_page_config(
    page_title="Waste Detection System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title("🚀 Waste Detection System")
st.markdown("Real-time waste detection using YOLOv8 AI model with Robot Helper")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is recommended for Streamlit Cloud to maintain high FPS
    return YOLO('yolov8n.pt')

model = load_model()
waste_types = ['bottle', 'cup', 'can', 'box', 'trash', 'bag', 'plastic', 'paper', 'cardboard']

# --- Session State Initialization ---
if 'waste_count' not in st.session_state:
    st.session_state.waste_count = {}
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'collected_items' not in st.session_state:
    st.session_state.collected_items = deque(maxlen=20)
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'fps_history' not in st.session_state:
    st.session_state.fps_history = deque(maxlen=30)

# --- Robot Animation Logic ---
def draw_robot_animation(frame, is_reaching, last_waste_position, animation_time, current_time):
    annotated_frame = frame.copy()
    robot_base = (80, annotated_frame.shape[0] - 80)
    shoulder = (robot_base[0] + 40, robot_base[1] - 60)
    collection_success = False
    
    if is_reaching and last_waste_position:
        elapsed = current_time - animation_time
        progress = min(elapsed / 1.2, 1.0)
        target = last_waste_position
        ease_progress = progress if progress < 0.5 else 1 - (1 - progress) ** 2
        
        elbow = (int(shoulder[0] + (target[0] - shoulder[0]) * ease_progress * 0.5),
                 int(shoulder[1] + (target[1] - shoulder[1]) * ease_progress * 0.4))
        hand = (int(shoulder[0] + (target[0] - shoulder[0]) * ease_progress),
                int(shoulder[1] + (target[1] - shoulder[1]) * ease_progress))
        
        # Robot Body
        cv2.rectangle(annotated_frame, (robot_base[0] - 25, robot_base[1] - 80), (robot_base[0] + 25, robot_base[1]), (50, 200, 255), -1)
        cv2.rectangle(annotated_frame, (robot_base[0] - 25, robot_base[1] - 80), (robot_base[0] + 25, robot_base[1]), (0, 255, 0), 3)
        cv2.circle(annotated_frame, (robot_base[0], robot_base[1] - 95), 15, (50, 200, 255), -1)
        
        # Arm segments
        cv2.line(annotated_frame, shoulder, elbow, (0, 255, 150), 4)
        cv2.line(annotated_frame, elbow, hand, (0, 200, 255), 4)
        
        # Status
        cv2.putText(annotated_frame, f"COLLECTING... {int(progress * 100)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if progress >= 1.0:
            collection_success = True
            return annotated_frame, False, None, 0, collection_success
        return annotated_frame, True, last_waste_position, animation_time, False
    else:
        # Idle Robot
        cv2.rectangle(annotated_frame, (robot_base[0] - 25, robot_base[1] - 80), (robot_base[0] + 25, robot_base[1]), (100, 150, 200), -1)
        cv2.circle(annotated_frame, (robot_base[0], robot_base[1] - 95), 15, (100, 150, 200), -1)
        cv2.putText(annotated_frame, "SCANNING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        return annotated_frame, False, None, 0, False

# --- WebRTC Video Processor ---
class VideoProcessor:
    def __init__(self):
        self.is_reaching = False
        self.last_waste_detection = None
        self.animation_time = 0
        self.conf_threshold = 0.45

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        results = model(img, conf=self.conf_threshold)
        annotated_frame = results[0].plot()
        
        closest_waste = None
        for detection in results[0].boxes:
            class_name = model.names[int(detection.cls[0])]
            if any(w in class_name.lower() for w in waste_types):
                conf = float(detection.conf[0])
                x1, y1, x2, y2 = detection.xyxy[0]
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if closest_waste is None or conf > closest_waste[2]:
                    closest_waste = (center, class_name, conf)

        if closest_waste and not self.is_reaching:
            self.is_reaching = True
            self.last_waste_detection = closest_waste[0]
            self.animation_time = current_time
            # Update counts (Note: we use a hack to update session state from this thread)
            st.session_state.total_detections += 1
            name = closest_waste[1]
            st.session_state.waste_count[name] = st.session_state.waste_count.get(name, 0) + 1
            st.session_state.collected_items.append({'name': name, 'time': datetime.now().strftime("%H:%M:%S")})

        annotated_frame, self.is_reaching, self.last_waste_detection, self.animation_time, _ = draw_robot_animation(
            annotated_frame, self.is_reaching, self.last_waste_detection, self.animation_time, current_time
        )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- UI Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Detection Confidence", 0.1, 1.0, 0.45, 0.05)

# --- Main Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Feed")
    ctx = webrtc_streamer(
        key="waste-detection",
        mode="transform",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )
    if ctx.video_processor:
        ctx.video_processor.conf_threshold = conf_thresh

with col2:
    st.subheader("📊 Real-time Stats")
    st.metric("✅ Total Collected", st.session_state.total_detections)
    if st.session_state.collected_items:
        st.write("**Recent Collections:**")
        for item in list(st.session_state.collected_items)[-5:]:
            st.write(f"• {item['name']} ({item['time']})")

# --- Bottom Row Info ---
bcol1, bcol2 = st.columns(2)
with bcol1:
    st.subheader("📈 Top Waste Types")
    if st.session_state.waste_count:
        for w_type, count in sorted(st.session_state.waste_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"**{w_type}**: {count}")
    else:
        st.write("No items collected yet.")

with bcol2:
    st.subheader("⏱️ Session Info")
    elapsed = time.time() - st.session_state.start_time
    st.write(f"⏱️ **Duration**: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

# --- Final Report Section ---
st.markdown("---")
st.subheader("📋 Session Report")

if st.session_state.total_detections > 0:
    waste_data = [{"Waste Type": k, "Collected": v, "Percentage": f"{(v/st.session_state.total_detections)*100:.1f}%"} 
                  for k, v in st.session_state.waste_count.items()]
    st.dataframe(waste_data, use_container_width=True)
    
    report_text = f"WASTE DETECTION REPORT\nTotal Collected: {st.session_state.total_detections}\n"
    for item in waste_data:
        report_text += f"{item['Waste Type']}: {item['Collected']}\n"
        
    st.download_button("📥 Download Report", data=report_text, file_name="report.txt")