import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

# --- CONFIGURATION ---
DATA_PATH = "../data/raw/train.csv"
MODEL_OUTPUT_DIR = "../models/my_custom_moderation_model"

# 1. LOAD DATA
print(f"Loading data from {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Cannot find {DATA_PATH}. Please download 'train.csv' from Kaggle and place it in data/raw/")

df = pd.read_csv(DATA_PATH)

# For the demo, we use 10,000 samples to keep training under 20 mins on your 3050 Ti.
# You can increase this to 50,000+ for better accuracy if you have time.
texts = df['comment_text'].tolist()[:10000]
labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.tolist()[:10000]

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 2. TOKENIZATION
print("Tokenizing data (this takes a moment)...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 3. DATASET CLASS
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_encodings, train_labels)
val_dataset = ToxicDataset(val_encodings, val_labels)

# 4. INITIALIZE MODEL
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# 5. TRAINING ARGUMENTS (Optimized for RTX 3050 Ti)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # Loops over the data
    per_device_train_batch_size=8,   # Low batch size for 4GB VRAM
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    fp16=True,                       # MIXED PRECISION (Critical for 3050 Ti)
    gradient_accumulation_steps=2,   # Virtual batch size = 16
    save_strategy="epoch",
    eval_strategy="epoch",
    dataloader_num_workers=0         # Windows compatibility
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 6. TRAIN & SAVE
print("Starting training on GPU...")
trainer.train()

print(f"Saving model to {MODEL_OUTPUT_DIR}...")
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print("DONE. You can now run text_moderation.py")