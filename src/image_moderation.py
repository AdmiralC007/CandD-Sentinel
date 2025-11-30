import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForImageClassification, AutoImageProcessor
import os
# Import your robust text system
from text_moderation import TwoStageModerator 

class ImageModerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("1. Loading NSFW Detector (Falconsai)...")
        # Specialized model for Nudity/Pornography detection
        try:
            self.nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
            self.nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(self.device)
            self.nsfw_enabled = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load NSFW model ({e}). Skipping visual NSFW check.")
            self.nsfw_enabled = False

        # --- NEW: Violence Detector ---
        print("1b. Loading Violence Detector (Jaranohaal)...")
        try:
            self.violence_processor = AutoImageProcessor.from_pretrained("jaranohaal/vit-base-violence-detection")
            self.violence_model = AutoModelForImageClassification.from_pretrained("jaranohaal/vit-base-violence-detection").to(self.device)
            self.violence_enabled = True
        except Exception as e:
            print(f"⚠️ Warning: Could not load Violence model ({e}). Skipping violence check.")
            self.violence_enabled = False
        # ------------------------------

        print("2. Loading Captioning Model (BLIP)...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        print("3. Loading Text Logic...")
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
            
        # Label 1 is usually 'nsfw' in this model, 0 is 'normal'
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
            # Use softmax to get probabilities
            probs = outputs.logits.softmax(dim=1)
            
            # Get the label with highest probability
            predicted_label_id = probs.argmax(-1).item()
            label_name = self.violence_model.config.id2label[predicted_label_id]
            confidence = probs[0][predicted_label_id].item()

        # Check if label is 'violence' and confidence is high (> 0.85)
        if label_name.lower() == 'violence' and confidence > 0.85:
            print(f"   🚨 Violence/Gore Detected! (Confidence: {confidence:.2f})")
            return True
        return False

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
            
            # --- CHECK 1: VISUAL NSFW DETECTOR ---
            if self.check_nsfw(raw_image):
                print(f"   👉 Final Verdict: ❌ UNSAFE (Visual Nudity)")
                return 1 
            
            # --- CHECK 1b: VISUAL VIOLENCE DETECTOR (New) ---
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