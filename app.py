import streamlit as st
import sys
import os
import gc  # Garbage Collection
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add 'src' to path
sys.path.append(os.path.abspath("src"))

# Import modules with Cloud compatibility
try:
    from text_moderation import TwoStageModerator
    from image_moderation import ImageModerator
    from video_moderation import VideoModerator
except ImportError:
    from src.text_moderation import TwoStageModerator
    from src.image_moderation import ImageModerator
    from src.video_moderation import VideoModerator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="C&D Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* Main Background */
        .stApp { background-color: #0E1117; color: #FAFAFA; padding-bottom: 80px; }
        
        /* Hide Decoration */
        header[data-testid="stHeader"] { background-color: #0E1117; z-index: 1; }
        .st-emotion-cache-12fmw85 { display: none; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Buttons */
        div.stButton > button {
            background-color: #262730; color: white; border: 1px solid #4B4B4B;
            border-radius: 8px; font-weight: 500; transition: all 0.3s ease;
        }
        div.stButton > button:hover { background-color: #FF4B4B; border-color: #FF4B4B; color: white; }
        
        /* Footer */
        .custom-footer {
            position: fixed; left: 0; bottom: 0; width: 100%; background-color: #161920;
            color: #808495; text-align: center; padding: 15px 0; font-size: 14px;
            font-family: 'Source Sans Pro', sans-serif; border-top: 1px solid #262730; z-index: 999;
        }
    </style>
    <div class="custom-footer">
        Made by <b>M Chaitanya</b> and <b>V Darpad Sai</b> | All Rights Reserved @ 2025
    </div>
""", unsafe_allow_html=True)

st.title("🛡️ C&D Sentinel")
st.caption("Enterprise-Grade Multimodal Content Moderation Pipeline")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Status")
    if os.getenv("GROQ_API_KEY"):
        st.success("● LLM Online (Groq)")
    else:
        st.error("● LLM Offline")
    
    st.markdown("### 🎚️ Sensitivity Settings")
    safe_th = st.slider("Safe Threshold", 0.0, 0.5, 0.2, 0.05)
    toxic_th = st.slider("Toxic Threshold", 0.5, 1.0, 0.9, 0.05)
    
    st.info(f"""
    **Decision Logic:**
    - **Score < {safe_th}**: ✅ Auto-Safe
    - **Score > {toxic_th}**: ❌ Auto-Toxic
    - **Between**: ⚠️ Ambiguous -> **LLM Check**
    """)
    
    if st.button("🧹 Force Clear RAM"):
        gc.collect()
        st.toast("Memory Released")

# --- LOAD MODELS (OPTIMIZED) ---

# 1. Keep Text Model Cached (Small ~250MB)
@st.cache_resource
def load_text_model():
    local_path = "models/my_custom_moderation_model"
    if os.path.exists(local_path):
        return TwoStageModerator(model_path=local_path)
    else:
        return TwoStageModerator(model_path="unitary/toxic-bert")

# 2. DO NOT CACHE Image/Video Models (Huge ~1GB+)
def load_image_model_ondemand():
    return ImageModerator()

def load_video_model_ondemand():
    return VideoModerator()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["💬 Text Analysis", "🖼️ Image Scan", "🎬 Video Audit"])

# === TAB 1: TEXT ===
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Text Engine")
        user_text = st.text_area("Input Content", height=150, placeholder="Type text here...")
        
        if st.button("Analyze Text", key="text_btn"):
            if user_text:
                text_bot = load_text_model()
                with st.spinner("Running Pipeline..."):
                    # Get raw metrics
                    tox_score = text_bot.stage_a_predict(user_text)
                    sent_label, _ = text_bot.get_sentiment(user_text)
                    
                    # Run Logic
                    verdict = text_bot.predict_verdict(user_text, safe_threshold=safe_th, toxic_threshold=toxic_th)
                
                # Logic Explanation
                logic_used = "Unknown"
                if tox_score > 0.97: logic_used = "Severe Toxicity Override"
                elif tox_score > toxic_th: logic_used = "High Confidence Threshold"
                elif tox_score < safe_th: logic_used = "Low Confidence Threshold"
                else: logic_used = "LLM Contextual Reasoning"

                if verdict == 1:
                    st.error(f"### ❌ VERDICT: UNSAFE")
                    st.write(f"**Logic:** {logic_used}")
                    st.write(f"**Score:** `{tox_score:.4f}` | **Sentiment:** `{sent_label}`")
                else:
                    st.success(f"### ✅ VERDICT: SAFE")
                    st.write(f"**Logic:** {logic_used}")
                    st.write(f"**Score:** `{tox_score:.4f}` | **Sentiment:** `{sent_label}`")
    
    with col2:
        st.info("**Pipeline Stages:**\n1. DistilBERT Toxicity Scan\n2. Sentiment Analysis\n3. Keyword Heuristics\n4. Llama-3 Reasoning (if needed)")

# === TAB 2: IMAGE ===
with tab2:
    st.subheader("Visual Content Analysis")
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, width=400)
        
        if st.button("Scan Image", key="img_btn"):
            with st.spinner("Loading AI Vision Model..."):
                img_bot = load_image_model_ondemand()
                temp_path = "temp_upload.jpg"
                image.save(temp_path)
                
                # Inject dynamic thresholds
                img_bot.text_moderator.predict_verdict = lambda t: load_text_model().predict_verdict(t, safe_th, toxic_th)
                
                verdict = img_bot.moderate_image(temp_path)
                
                if os.path.exists(temp_path): os.remove(temp_path)
                
                if verdict == 1: st.error("❌ **UNSAFE IMAGE** (Violence/Hate Detected)")
                else: st.success("✅ **SAFE IMAGE**")
                
                del img_bot
                gc.collect()

# === TAB 3: VIDEO ===
with tab3:
    st.subheader("Multimodal Video Audit")
    uploaded_video = st.file_uploader("Upload MP4", type=["mp4", "mov"])
    
    if uploaded_video:
        tfile = "temp_video_upload.mp4"
        with open(tfile, 'wb') as f:
            f.write(uploaded_video.read())
        st.video(tfile)
        
        if st.button("Start Audit", key="vid_btn"):
            st.info("Initializing Multimodal Pipeline... (Please wait)")
            progress_bar = st.progress(0)
            
            vid_bot = load_video_model_ondemand()
            
            # Inject thresholds
            vid_bot.text_bot.predict_verdict = lambda t: load_text_model().predict_verdict(t, safe_th, toxic_th)
            vid_bot.image_bot.text_moderator.predict_verdict = lambda t: load_text_model().predict_verdict(t, safe_th, toxic_th)

            # 1. Audio
            transcript = vid_bot.transcribe_audio(tfile)
            audio_safe = True
            if transcript:
                if vid_bot.text_bot.predict_verdict(transcript) == 1: audio_safe = False
            progress_bar.progress(40)
            
            # 2. Visual Check
            import cv2
            cap = cv2.VideoCapture(tfile)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = int(fps * 3) 
            visual_safe = True
            curr = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if curr % frame_skip == 0:
                    cv2.imwrite("temp_frame_ui.jpg", frame)
                    if vid_bot.image_bot.moderate_image("temp_frame_ui.jpg") == 1:
                        visual_safe = False
                        break
                    gc.collect()
                curr += 1
            cap.release()
            progress_bar.progress(100)
            
            # Report
            col_a, col_v = st.columns(2)
            with col_a:
                if audio_safe: 
                    st.success("🔊 Audio: Safe")
                else: 
                    st.error("🔊 Audio: TOXIC")
                    with st.expander("See Transcript"):
                        st.write(transcript if transcript else "No speech detected")

            with col_v:
                if visual_safe: st.success("🖼️ Visuals: Safe")
                else: st.error("🖼️ Visuals: UNSAFE")
            
            del vid_bot
            gc.collect()