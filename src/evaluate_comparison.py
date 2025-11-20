import pandas as pd
import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# Import BOTH moderators
from image_moderation import ImageModerator as BLIPBot
from image_moderation_2 import CLIPModerator as CLIPBot

# --- CONFIGURATION ---
DATASET_JSONL = "../data/raw/hateful_memes/train.jsonl"
IMAGE_FOLDER = "../data/raw/hateful_memes/img/"
TEST_SIZE = 50 # Number of images to test

def run_comparison():
    print("🚀 Starting A/B Comparison: BLIP vs. CLIP")
    
    # 1. Load Data
    data = []
    with open(DATASET_JSONL, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data).head(TEST_SIZE)
    ground_truth = df['label'].tolist()
    
    # 2. Initialize Models
    print("   Loading BLIP Model...")
    blip_bot = BLIPBot()
    print("   Loading CLIP Model...")
    clip_bot = CLIPBot()
    
    blip_preds = []
    clip_preds = []
    
    # 3. Run Tests
    print("\n   Processing Images...")
    for index, row in tqdm(df.iterrows(), total=TEST_SIZE):
        img_path = os.path.join("../data/raw/hateful_memes", row['img'])
        
        # Get Prediction from BLIP
        # (We suppress print output inside the loop for cleaner progress bar)
        # Note: In a real run, you might want to redirect stdout to quiet it
        p1 = blip_bot.moderate_image(img_path)
        blip_preds.append(p1)
        
        # Get Prediction from CLIP
        p2 = clip_bot.moderate_image(img_path)
        clip_preds.append(p2)

    # 4. Calculate & Compare
    acc_blip = accuracy_score(ground_truth, blip_preds)
    acc_clip = accuracy_score(ground_truth, clip_preds)
    
    print("\n" + "="*40)
    print(" 🏆 FINAL SHOWDOWN RESULTS")
    print("="*40)
    print(f"   BLIP (Method 1) Accuracy: {acc_blip*100:.2f}%")
    print(f"   CLIP (Method 2) Accuracy: {acc_clip*100:.2f}%")
    
    winner = "CLIP" if acc_clip > acc_blip else "BLIP"
    if acc_clip == acc_blip: winner = "Tie"
    
    print(f"\n   👑 Winner: {winner}")
    print("="*40)

if __name__ == "__main__":
    run_comparison()