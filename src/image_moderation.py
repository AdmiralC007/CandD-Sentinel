import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
# Import your robust text system
from text_moderation import TwoStageModerator 

class ImageModerator:
    def __init__(self):
        print("1. Loading Image Captioning Model (BLIP)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"   BLIP Model loaded on {self.device.upper()}.")

        print("2. Loading Text Moderation Pipeline...")
        self.text_moderator = TwoStageModerator() 
        print("   Ready for Multimodal Analysis.\n")

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
            
            # B. Generate Caption
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=50, min_new_tokens=10)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            print(f"   👀 Generated Caption: \"{caption}\"")
            
            # C. Pass to Text Moderator (Centralized Logic)
            # This ensures we use the EXACT same keywords/sentiment logic as the text system
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
    
    # Create dummy images if they don't exist
    if not os.path.exists("test_safe.jpg"):
        Image.new('RGB', (100, 100), color='blue').save("test_safe.jpg")
        Image.new('RGB', (100, 100), color='red').save("test_unsafe.jpg")

    # Test Returns
    print("\n--- Testing Return Values ---")
    bot.moderate_image("test_safe.jpg")
    bot.moderate_image("test_unsafe.jpg")