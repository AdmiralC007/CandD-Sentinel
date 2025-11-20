import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

class CLIPModerator:
    def __init__(self):
        print("1. Loading CLIP Model (Zero-Shot Classifier)...")
        # We use the "base" version which is fast and accurate
        # Source: OpenAI CLIP (ViT-B/32)
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        
        # Move to GPU (RTX 3050 Ti)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"   CLIP Model loaded on {self.device.upper()}.")

        # Define the concepts we want to check against
        # This is the "Zero-Shot" magic. We define what "Safe" and "Unsafe" look like.
        self.labels = [
            "a photo of something safe and normal", 
            "a photo of violence or physical harm", 
            "a photo of a gun or weapon", 
            "a photo of blood or gore", 
            "a photo of hate speech or offensive content"
        ]
        print(f"   Labels Defined: {self.labels}\n")

    def moderate_image(self, image_path):
        """
        Analyzes an image using CLIP.
        Returns:
        0 = SAFE
        1 = UNSAFE / TOXIC
        """
        print(f"\n" + "="*40)
        print(f" 👁️  Analyzing Image (CLIP): {image_path}")
        print(f"="*40)
        
        try:
            if not os.path.exists(image_path):
                print(f"❌ Error: Image not found at {image_path}")
                return 0 

            # A. Load Image
            image = Image.open(image_path).convert('RGB')
            
            # B. Process Inputs (Image + Text Labels)
            inputs = self.processor(
                text=self.labels, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # C. Run Model
            outputs = self.model(**inputs)
            
            # D. Get Probabilities
            # The model calculates similarity scores (logits) for each label
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1) # Convert to %
            
            # Move to CPU to read values
            probs_list = probs.detach().cpu().numpy()[0]
            
            # E. Interpret Results
            # Index 0 is "safe", Indices 1-4 are "unsafe" variants
            safe_score = probs_list[0]
            unsafe_score = sum(probs_list[1:]) # Sum of all unsafe probabilities
            
            # Find the top category for display
            highest_prob_index = probs_list.argmax()
            top_label = self.labels[highest_prob_index]
            top_score = probs_list[highest_prob_index]
            
            print(f"   Top Match: '{top_label}' ({top_score*100:.2f}%)")
            print(f"   --------------------------------")
            print(f"   Safe Confidence:   {safe_score*100:.2f}%")
            print(f"   Unsafe Confidence: {unsafe_score*100:.2f}%")
            
            # F. Decision Logic
            # If Unsafe confidence > Safe confidence, flag it.
            if unsafe_score > safe_score:
                print(f"   👉 Final Verdict: ❌ UNSAFE")
                return 1
            else:
                print(f"   👉 Final Verdict: ✅ SAFE")
                return 0

        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            return 0

# --- TEST EXECUTION ---
if __name__ == "__main__":
    bot = CLIPModerator()
    
    # Test with the same images you used for BLIP
    if os.path.exists("test_safe.jpg"):
        bot.moderate_image("test_safe.jpg")
    if os.path.exists("test_unsafe.jpg"):
        bot.moderate_image("test_unsafe.jpg")