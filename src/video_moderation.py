import cv2
import os
import time
from groq import Groq
from dotenv import load_dotenv

# Handle MoviePy version differences safely
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# Import your existing modules
from image_moderation import ImageModerator
from text_moderation import TwoStageModerator

load_dotenv()

class VideoModerator:
    def __init__(self):
        print("1. Initializing Multimodal Video Pipeline...")
        
        # A. Visual Engine (BLIP + Text Pipeline)
        self.image_bot = ImageModerator()
        
        # B. Audio Engine (Groq Whisper)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ WARNING: GROQ_API_KEY not found. Audio analysis will fail.")
        self.groq_client = Groq(api_key=api_key)
        
        # C. Reasoning Engine (for Audio text)
        self.text_bot = TwoStageModerator()
        
        print("   Video + Audio Analysis Ready.\n")

    def transcribe_audio(self, video_path):
        """Extracts audio and uses Groq Whisper to convert to text"""
        print("\n   🎤 Extracting & Transcribing Audio...")
        try:
            # 1. Extract Audio using MoviePy
            clip = VideoFileClip(video_path)
            
            if clip.audio is None:
                print("      (No audio track found)")
                clip.close()
                return ""
            
            audio_path = "temp_audio.mp3"
            
            # Use mp3 with specific codec to avoid FFMPEG errors
            clip.audio.write_audiofile(audio_path, codec="libmp3lame", logger=None)
            
            clip.close() # Important: Close file handle to release memory
            
            # 2. Send to Groq Whisper
            with open(audio_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(audio_path, file.read()),
                    
                    # --- FIX: Updated to the new Turbo model ---
                    model="whisper-large-v3-turbo", 
                    # -------------------------------------------
                    
                    response_format="text"
                )
            
            # Cleanup temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            print(f"      🗣️  Spoken Text: \"{transcription.strip()[:100]}...\"")
            return transcription
            
        except Exception as e:
            print(f"      ❌ Audio Error: {e}")
            return ""

    def moderate_video(self, video_path, frame_interval=2):
        print(f"\n" + "="*50)
        print(f" 🎬 ANALYZING VIDEO (VISUAL + AUDIO): {video_path}")
        print(f"="*50)

        if not os.path.exists(video_path):
            print(f"❌ Error: File not found at {video_path}")
            return

        # --- PHASE 1: AUDIO ANALYSIS ---
        spoken_text = self.transcribe_audio(video_path)
        audio_verdict = 0
        
        if spoken_text:
            print("   🧠 Analyzing Spoken Content...")
            # Reuse your robust Text Logic
            audio_verdict = self.text_bot.predict_verdict(spoken_text)
            
            if audio_verdict == 1:
                print("      👉 AUDIO VERDICT: ❌ TOXIC (Hate speech/threats detected)")
            else:
                print("      👉 AUDIO VERDICT: ✅ SAFE")
        else:
            print("      (Skipping Audio Analysis)")

        # --- PHASE 2: VISUAL ANALYSIS ---
        print("\n   👁️  Starting Visual Frame Analysis...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            print("❌ Error: Could not read video frames.")
            return

        frames_to_skip = int(fps * frame_interval)
        current_frame = 0
        visual_unsafe_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if current_frame % frames_to_skip == 0:
                timestamp = current_frame / fps
                
                # Save temp frame
                cv2.imwrite("temp_frame.jpg", frame)
                
                # Check Visuals (Returns 0 or 1)
                is_toxic = self.image_bot.moderate_image("temp_frame.jpg")
                
                if is_toxic == 1:
                    print(f"      ⏱️  {timestamp:.1f}s: 🚨 VISUAL THREAT DETECTED")
                    visual_unsafe_count += 1
                else:
                    # Optional: Print 'Safe' for each frame to show progress
                    # print(f"      ⏱️  {timestamp:.1f}s: ✅ Frame Safe")
                    pass

            current_frame += 1
            
        cap.release()
        if os.path.exists("temp_frame.jpg"): os.remove("temp_frame.jpg")

        # --- FINAL VERDICT ---
        print("\n" + "="*50)
        print(" 🏁 FINAL MULTIMODAL REPORT")
        print("="*50)
        
        # Audio Status
        status_icon = "❌ TOXIC" if audio_verdict == 1 else "✅ SAFE"
        print(f"   🔊 Audio Status:  {status_icon}")
             
        # Visual Status
        if visual_unsafe_count > 0:
            print(f"   🖼️  Visual Status: ❌ UNSAFE ({visual_unsafe_count} flagged frames)")
        else:
            print(f"   🖼️  Visual Status: ✅ SAFE")

        print("-" * 30)
        
        # OR Logic: If EITHER is toxic, the whole video is unsafe
        if audio_verdict == 1 or visual_unsafe_count > 0:
            print(f"   👉 FINAL RESULT: ❌ UNSAFE VIDEO")
        else:
            print(f"   👉 FINAL RESULT: ✅ SAFE VIDEO")
        print("="*50 + "\n")

if __name__ == "__main__":
    bot = VideoModerator()
    
    # Ensure this path matches your file
    test_video = "../data/raw/test_videos/movie_scene.mp4"
    
    if os.path.exists(test_video):
        bot.moderate_video(test_video)
    else:
        print(f"Please check the file path: {test_video}")