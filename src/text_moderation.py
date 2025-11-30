import torch
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from groq import Groq

# 1. Load API Keys
load_dotenv()

class TwoStageModerator:
    def __init__(self, model_path=None):
        # --- DYNAMIC PATH RESOLUTION ---
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "models", "my_custom_moderation_model")
            model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            print(f"⚠️ Local model not found at: {model_path}")
            print("   Switching to Hugging Face default: unitary/toxic-bert")
            model_path = "unitary/toxic-bert"

        # --- PHASE 1: LOAD TOXICITY MODEL (Stage A) ---
        print(f"Loading Stage A Model from: {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            print("✅ Stage A (Toxicity) Loaded.")
        except OSError as e:
            print(f"❌ Error loading model: {e}")
            exit()

        # --- PHASE 1.5: LOAD SENTIMENT MODEL ---
        print("Loading Sentiment Analyzer for Context Checking...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Sentiment Filter Loaded.")

        # --- PHASE 2: SETUP LLM CLIENT (GROQ) ---
        self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def stage_a_predict(self, text):
        """Returns Toxicity Score (0.0 - 1.0)"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.sigmoid(outputs.logits).numpy()[0]
        return np.max(probs)

    def get_sentiment(self, text):
        """Returns 'POSITIVE' or 'NEGATIVE'"""
        safe_text = text[:512] if len(text) > 512 else text
        result = self.sentiment_analyzer(safe_text, truncation=True, max_length=512)[0]
        return result['label'], result['score']

    def stage_b_predict_llm(self, text):
        """Stage B: Groq LLM"""
        print(f"   [Stage B] 🧠 Sending to LLM: '{text[:50]}...'")
        
        system_prompt = """
        You are a Content Moderation AI. Your task is to classify text as SAFE or UNSAFE.
        
        Follow this logic strictly in order:
        
        1. **ZERO TOLERANCE (UNSAFE):** - If the text contains direct, realistic threats of violence (e.g., "I will kill you", "I'm going to find you", "die now").
           - If it encourages self-harm or terrorism.
           - **VERDICT:** UNSAFE (even if it sounds like a movie quote).
           
        2. **CONTEXTUAL EXCEPTIONS (SAFE):**
           - If the text is clearly **Gaming Slang** (e.g., "kill the boss", "shoot him", "I'm dead", "pwned").
           - If it is **Satire/Humor** (e.g., "I'm going to kill it on the dancefloor", "Roast battle").
           - If it is **Song Lyrics** or **Fictional Storytelling** without directing hate at a real person.
           - **VERDICT:** SAFE.

        3. **DEFAULT:**
           - If it is hateful bullying or harassment that doesn't fit the exceptions above, classify as UNSAFE.

        4. **CODED LANGUAGE:**
           - Watch for double meanings. If a word like "mudding" or "grooming" is used in a context that targets a specific group (like LGBTQ+), classify as UNSAFE.
        
        Format: Verdict: [SAFE/UNSAFE] Reasoning: [One sentence]
        """
        
        try:
            chat_completion = self.llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze: '{text[:4000]}'"}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0, 
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _check_keywords(self, text):
        """Checks for safe context keywords."""
        safe_contexts = [
            # Meta-commentary
            "kidding", "joke", "satire", "sarcasm", "jk", "/s", "literally", "context",
            
            # Gaming (Specific terms are okay)
            "game", "play", "round", "match", "server", "bot", "boss", "level", "quest", 
            
            # Music/Performance
            "song", "lyrics", "music", "track", "dance", "dancefloor", "stage",
            
            # Positive Slang
            "killing it", "sick", "insane", "beast", "fire"
            
            # DELETED: "movie", "film", "scene", "character", "story"
            # These are the culprits letting the bloody images through!
        ]
        text_lower = text.lower()
        for word in safe_contexts:
            if word in text_lower:
                return True, word
        return False, None

    def _analyze_single_chunk(self, text, safe_th, toxic_th, verbose=False):
        """Internal logic for one chunk."""
        tox_score = self.stage_a_predict(text)
        if verbose: print(f"   [Stage A] Toxicity: {tox_score:.4f}")

        # 1. SEVERE OVERRIDE (> 0.97)
        if tox_score > 0.97:
             if verbose: print("   Result: ❌ TOXIC (Severe Toxicity Override)")
             return 1

        # 2. Context Checks
        sent_label, sent_conf = self.get_sentiment(text)
        if verbose: print(f"   [Context] Sentiment: {sent_label} ({sent_conf:.4f})")

        has_context, found_word = self._check_keywords(text)
        if verbose and has_context:
            print(f"   [Context] Keyword Detected: Yes ('{found_word}')")

        is_ambiguous = (sent_label == "POSITIVE") or has_context
        
        # 3. Decision
        if tox_score > toxic_th and not is_ambiguous:
            if verbose: print(f"   Result: ❌ TOXIC (Score > {toxic_th})")
            return 1 
        elif tox_score < safe_th:
            if verbose: print(f"   Result: ✅ SAFE (Score < {safe_th})")
            return 0 
        else:
            if verbose: 
                reason = "Sentiment/Keyword Conflict" if is_ambiguous else f"Medium Confidence"
                print(f"   Result: ⚠️ AMBIGUOUS ({reason}) -> Calling LLM...")
            
            decision = self.stage_b_predict_llm(text)
            if verbose: print(f"   LLM Verdict: {decision}")
            
            # --- BUG FIX: Check UNSAFE *before* checking SAFE ---
            # The string "UNSAFE" contains "SAFE", so the order matters!
            decision_upper = decision.upper()
            if "UNSAFE" in decision_upper:
                return 1
            elif "SAFE" in decision_upper:
                return 0
            else:
                # Fallback for unclear responses
                return 1

    def predict_verdict(self, text, safe_threshold=0.20, toxic_threshold=0.90, verbose=True):
        """
        Helper for Image/Video/UI modules.
        HANDLES BATCHING to ensure NO truncation of toxic parts.
        """
        CHUNK_SIZE = 1000 
        
        # Case 1: Short Text
        if len(text) < CHUNK_SIZE:
            return self._analyze_single_chunk(text, safe_threshold, toxic_threshold, verbose=verbose)
        
        # Case 2: Long Text (Batching)
        if verbose: print(f"   [Batching] Text length {len(text)} > {CHUNK_SIZE}. Processing in chunks...")
        
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        
        for i, chunk in enumerate(chunks):
            verdict = self._analyze_single_chunk(chunk, safe_threshold, toxic_threshold, verbose=False)
            if verdict == 1:
                if verbose: print(f"   [Batching] 🚨 Toxic content found in Chunk #{i+1}. Marking Full Content as TOXIC.")
                return 1 
        
        if verbose: print("   [Batching] All chunks passed. Content is Safe.")
        return 0

    def moderate(self, text):
        """Interactive demo mode"""
        print(f"\nInput: \"{text[:100]}...\"")
        self.predict_verdict(text, verbose=True)

# --- INTERACTIVE TEST EXECUTION ---
if __name__ == "__main__":
    bot = TwoStageModerator()
    print("\n" + "="*50)
    print(" 🛡️  TWO-STAGE MODERATION SYSTEM READY")
    print(" Type a sentence to test (or 'exit' to quit)")
    print("="*50 + "\n")

    while True:
        user_input = input(">> Enter text: ")
        if user_input.lower() in ["exit", "quit"]: break
        bot.moderate(user_input)
        print("-" * 50)