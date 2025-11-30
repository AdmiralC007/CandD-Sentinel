import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForImageClassification, AutoImageProcessor
import os
import numpy as np
from text_moderation import TwoStageModerator 

# --- CLOUD SAFETY FIX ---
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("⚠️ Face Recognition library not found. Watchlist feature disabled.")

class ImageModerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- 1. NSFW Detector ---
        print("1. Loading NSFW Detector (Falconsai)...")
        try:
            self.nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
            self.nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(self.device)
            self.nsfw_enabled = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load NSFW model ({e}). Skipping visual NSFW check.")
            self.nsfw_enabled = False

        # --- 2. Violence Detector ---
        print("1b. Loading Violence Detector (Jaranohaal)...")
        try:
            self.violence_processor = AutoImageProcessor.from_pretrained("jaranohaal/vit-base-violence-detection")
            self.violence_model = AutoModelForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection").to(self.device)
            self.violence_enabled = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load Violence model ({e}). Skipping violence check.")
            self.violence_enabled = False
            
        # --- 3. Face Watchlist (Identity Recognition) ---
        print("2. Loading Face Watchlist...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        if FACE_REC_AVAILABLE:
            # Try standard paths
            potential_paths = [
                "../data/watchlist",
                "data/watchlist",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "watchlist")
            ]
            
            watchlist_path = None
            for p in potential_paths:
                if os.path.exists(p):
                    watchlist_path = p
                    break
                
            if watchlist_path:
                for filename in os.listdir(watchlist_path):
                    if filename.endswith((".jpg", ".png", ".jpeg")):
                        try:
                            path = os.path.join(watchlist_path, filename)
                            img = face_recognition.load_image_file(path)
                            encodings = face_recognition.face_encodings(img)
                            
                            if encodings:
                                self.known_face_encodings.append(encodings[0])
                                self.known_face_names.append(os.path.splitext(filename)[0])
                                print(f"   - Loaded banned face: {filename}")
                        except Exception as e:
                            print(f"   - Error loading {filename}: {e}")
            else:
                print(f"   ⚠️ Watchlist folder not found (checked common paths).")
        else:
            print("   ⚠️ Skipping Watchlist loading (Library missing).")

        # --- 4. BLIP Captioning ---
        print("3. Loading Captioning Model (BLIP)...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        # --- 5. Text Logic ---
        print("4. Loading Text Logic...")
        self.text_moderator = TwoStageModerator() 
        print("   Ready for Multimodal Analysis.\n")

    def check_nsfw(self, image):
        """Returns True if image is explicit/pornographic"""
        if not self.nsfw_enabled: return False
        
        inputs = self.nsfw_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.nsfw_model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            
        label_name = self.nsfw_model.config.id2label[predicted_label]
        
        if label_name.lower() == 'nsfw':
            print("   🚨 NSFW/Nudity Detected by Visual Classifier!")
            return True
        return False

    def check_violence(self, image):
        """Returns True if image depicts physical violence/gore"""
        if not self.violence_enabled: return False
        
        inputs = self.violence_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.violence_model(**inputs)
            probs = outputs.logits.softmax(dim=1)
            predicted_label_id = probs.argmax(-1).item()
            label_name = self.violence_model.config.id2label[predicted_label_id]
            confidence = probs[0][predicted_label_id].item()

        if label_name.lower() == 'violence' and confidence > 0.85:
            print(f"   🚨 Violence/Gore Detected! (Confidence: {confidence:.2f})")
            return True
        return False

    def check_watchlist(self, image_path):
        """Checks if any face in the image matches the banned list"""
        if not FACE_REC_AVAILABLE or not self.known_face_encodings:
            return False, None
        
        try:
            unknown_image = face_recognition.load_image_file(image_path)
            unknown_face_encodings = face_recognition.face_encodings(unknown_image)

            for unknown_face_encoding in unknown_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, unknown_face_encoding, tolerance=0.6)
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    print(f"   🚨 IDENTITY MATCH DETECTED: {name}")
                    return True, name
            
            return False, None
        except Exception as e:
            print(f"   ⚠️ Face check error: {e}")
            return False, None

    def moderate_image(self, image_path):
        """
        Analyzes an image and returns:
        0 = SAFE
        1 = UNSAFE / TOXIC
        """
        print(f"\n" + "="*40)
        print(f" Analyzing Image: {image_path}")
        print(f"="*40)
        
        try:
            if not os.path.exists(image_path):
                print(f"❌ Error: Image not found at {image_path}")
                return 0 

            # A. Load Image
            raw_image = Image.open(image_path).convert('RGB')
            
            # --- CHECK 0: WATCHLIST (Identity Check - Cloud Safe) ---
            is_banned, name = self.check_watchlist(image_path)
            if is_banned:
                print(f"   👉 Final Verdict: ❌ UNSAFE (Banned Individual: {name})")
                return 1

            # --- CHECK 1: VISUAL NSFW DETECTOR ---
            if self.check_nsfw(raw_image):
                print(f"   👉 Final Verdict: ❌ UNSAFE (Visual Nudity)")
                return 1 
            
            # --- CHECK 1b: VISUAL VIOLENCE DETECTOR ---
            if self.check_violence(raw_image):
                print(f"   👉 Final Verdict: ❌ UNSAFE (Visual Violence)")
                return 1
            
            # --- CHECK 2: SEMANTIC CAPTIONING (BLIP) ---
            inputs = self.blip_processor(raw_image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_new_tokens=50, min_new_tokens=10)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            print(f"   👀 Generated Caption: \"{caption}\"")
            
            # --- CHECK 3: TEXT LOGIC ---
            verdict = self.text_moderator.predict_verdict(caption)
            
            if verdict == 1:
                print(f"   👉 Final Verdict: ❌ TOXIC")
                return 1
            else:
                print(f"   👉 Final Verdict: ✅ SAFE")
                return 0

        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            return 0

# --- TEST EXECUTION ---
if __name__ == "__main__":
    bot = ImageModerator()
    
    if not os.path.exists("test_safe.jpg"):
        Image.new('RGB', (100, 100), color='blue').save("test_safe.jpg")
        Image.new('RGB', (100, 100), color='red').save("test_unsafe.jpg")

    print("\n--- Testing Return Values ---")
    bot.moderate_image("test_safe.jpg")
    bot.moderate_image("test_unsafe.jpg")