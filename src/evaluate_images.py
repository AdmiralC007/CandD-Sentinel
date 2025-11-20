import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm  # Progress bar
from image_moderation import ImageModerator

# --- CONFIGURATION ---
# Point this to your extracted files
# Note: JSONL file is usually inside the 'hateful_memes' folder
DATASET_JSONL = "../data/raw/hateful_memes/train.jsonl" 
IMAGE_FOLDER = "../data/raw/hateful_memes/img/"

def run_evaluation():
    print("🚀 Starting Full Evaluation Pipeline (Hateful Memes Dataset)...")
    
    # 1. Load Dataset (JSONL Handling)
    if not os.path.exists(DATASET_JSONL):
        print(f"❌ Error: Dataset file not found at {DATASET_JSONL}")
        print("   Please check the path. It should point to 'train.jsonl'.")
        return

    print("   Loading metadata from JSONL...")
    data = []
    with open(DATASET_JSONL, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    
    # The Hateful Memes dataset usually has columns: 'id', 'img', 'label', 'text'
    # We only need 'img' (filename) and 'label' (0=safe, 1=hateful)
    
    # --- LIMIT FOR SPEED ---
    # Running all 8,500+ images will take hours.
    # Let's run the first 50 images for your immediate test.
    df = df.head(50) 
    print(f"   Loaded {len(df)} images for testing.")

    # 2. Initialize Model
    bot = ImageModerator()
    
    predictions = []
    ground_truth = df['label'].tolist() 

    # 3. Loop through images
    print("   Processing images...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # The 'img' column usually looks like "img/01234.png"
        # We need to combine it with our base folder correctly
        # Since 'img' inside JSONL often includes the "img/" prefix, we check:
        img_rel_path = row['img']
        
        # Handle path joining carefully
        if img_rel_path.startswith("img/"):
            # If JSON says "img/01234.png", and IMAGE_FOLDER points to ".../img/", 
            # we need to be careful not to double "img/img".
            # Best approach: Point IMAGE_FOLDER to the PARENT of 'img' folder
            full_path = os.path.join("../data/raw/hateful_memes", img_rel_path)
        else:
            full_path = os.path.join(IMAGE_FOLDER, img_rel_path)
        
        # Get Prediction (0 = Safe, 1 = Unsafe)
        # Note: This calls the function that returns 0 or 1 (ensure you updated image_moderation.py!)
        pred = bot.moderate_image(full_path)
        predictions.append(pred)

    # 4. Calculate Metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)

    print("\n" + "="*40)
    print(" 📊 EVALUATION RESULTS")
    print("="*40)
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print("-" * 40)
    
    # Detailed Report
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=["Safe", "Unsafe"]))
    
    # 5. Save Results
    df['predicted'] = predictions
    df.to_csv("../data/evaluation_results.csv", index=False)
    print("\n✅ Detailed results saved to data/evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()