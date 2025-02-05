from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import numpy as np
from time import time

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
        self.cache_dir = os.getenv('CACHE_DIR', '.cache')
        self.max_length = int(os.getenv('MAX_LENGTH', '512'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Map label indices to sentiment labels
        self.id2label = {
            0: "negative",
            1: "positive"
        }

    def _preprocess_text(self, text: str) -> Dict:
        """Tokenize and prepare text for model input."""
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )

    def _get_confidence_scores(self, logits: torch.Tensor) -> Dict[str, float]:
        """Convert model logits to confidence scores."""
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        return {
            "negative": float(probs[0]),
            "positive": float(probs[1]),
            "neutral": float(max(1 - abs(probs[1] - probs[0]), 0))  # Neutral score based on confidence difference
        }

    def analyze_text(self, text: str, include_confidence_scores: bool = False) -> Dict:
        """Analyze the sentiment of a given text."""
        start_time = time()
        
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
        
        # Preprocess text
        inputs = self._preprocess_text(text)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predicted label
        predicted_label_id = torch.argmax(logits, dim=1).item()
        sentiment = self.id2label[predicted_label_id]
        
        # Prepare response
        response = {
            "sentiment": sentiment,
            "processing_time": time() - start_time
        }
        
        # Add confidence scores if requested
        if include_confidence_scores:
            response["confidence_scores"] = self._get_confidence_scores(logits)
        
        return response

    def analyze_batch(self, texts: List[str], include_confidence_scores: bool = False) -> List[Dict]:
        """Analyze sentiment for a batch of texts."""
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = [
                self.analyze_text(text, include_confidence_scores)
                for text in batch_texts
            ]
            results.extend(batch_results)
        
        return results
