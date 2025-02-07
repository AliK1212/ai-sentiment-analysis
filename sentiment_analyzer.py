from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import numpy as np
from time import time
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        # Load multiple models for ensemble approach
        self.models = {
            'distilbert': {
                'name': os.getenv('DISTILBERT_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english'),
                'model': None,
                'tokenizer': None,
                'weight': float(os.getenv('DISTILBERT_WEIGHT', '0.3'))
            },
            'roberta': {
                'name': os.getenv('ROBERTA_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                'model': None,
                'tokenizer': None,
                'weight': float(os.getenv('ROBERTA_WEIGHT', '0.3'))
            }
        }
        
        self.cache_dir = os.getenv('CACHE_DIR', '.cache')
        self.max_length = int(os.getenv('MAX_LENGTH', '512'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        
        # Initialize models and tokenizers
        for model_key, model_info in self.models.items():
            try:
                model_info['tokenizer'] = AutoTokenizer.from_pretrained(
                    model_info['name'],
                    cache_dir=self.cache_dir
                )
                model_info['model'] = AutoModelForSequenceClassification.from_pretrained(
                    model_info['name'],
                    cache_dir=self.cache_dir
                )
                model_info['model'].eval()
            except Exception as e:
                print(f"Error loading {model_key} model: {str(e)}")
                # Fallback to DistilBERT if other models fail
                if model_key != 'distilbert':
                    self.models[model_key]['weight'] = 0
                    self.models['distilbert']['weight'] = 1.0
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        self.vader_weight = float(os.getenv('VADER_WEIGHT', '0.2'))
        
        # Initialize TextBlob weight
        self.textblob_weight = float(os.getenv('TEXTBLOB_WEIGHT', '0.2'))
        
        # Normalize weights in case some models failed to load
        total_weight = sum(m['weight'] for m in self.models.values()) + self.vader_weight + self.textblob_weight
        if total_weight != 1.0:
            factor = 1.0 / total_weight
            for model_info in self.models.values():
                model_info['weight'] *= factor
            self.vader_weight *= factor
            self.textblob_weight *= factor
        
        # Map label indices to sentiment labels
        self.id2label = {
            0: "negative",
            1: "positive"
        }

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        if not text:
            return text
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s:;)(><]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def _get_ensemble_prediction(self, text: str) -> Dict:
        """Get predictions from multiple models and combine them."""
        predictions = {}
        
        # Get transformer model predictions
        for model_key, model_info in self.models.items():
            if model_info['weight'] > 0:
                try:
                    inputs = model_info['tokenizer'](
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        outputs = model_info['model'](**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                        
                        if model_key == 'roberta':
                            predictions[model_key] = {
                                "negative": float(probs[0]),
                                "positive": float(probs[2])  # RoBERTa has 3 classes
                            }
                        else:
                            predictions[model_key] = {
                                "negative": float(probs[0]),
                                "positive": float(probs[1])
                            }
                except Exception as e:
                    print(f"Error in {model_key} prediction: {str(e)}")
                    model_info['weight'] = 0
        
        # Get VADER prediction if weight > 0
        if self.vader_weight > 0:
            try:
                vader_scores = self.vader.polarity_scores(text)
                predictions['vader'] = {
                    "negative": vader_scores['neg'],
                    "positive": vader_scores['pos']
                }
            except Exception as e:
                print(f"Error in VADER prediction: {str(e)}")
                self.vader_weight = 0
        
        # Get TextBlob prediction if weight > 0
        if self.textblob_weight > 0:
            try:
                blob = TextBlob(text)
                textblob_polarity = (blob.sentiment.polarity + 1) / 2  # Normalize to 0-1
                predictions['textblob'] = {
                    "negative": 1 - textblob_polarity,
                    "positive": textblob_polarity
                }
            except Exception as e:
                print(f"Error in TextBlob prediction: {str(e)}")
                self.textblob_weight = 0
        
        # Combine predictions with weighted average
        final_scores = {"positive": 0.0, "negative": 0.0}
        total_weight = 0.0
        
        for model_key, model_info in self.models.items():
            if model_key in predictions and model_info['weight'] > 0:
                weight = model_info['weight']
                final_scores["negative"] += predictions[model_key]["negative"] * weight
                final_scores["positive"] += predictions[model_key]["positive"] * weight
                total_weight += weight
        
        if 'vader' in predictions and self.vader_weight > 0:
            final_scores["negative"] += predictions['vader']["negative"] * self.vader_weight
            final_scores["positive"] += predictions['vader']["positive"] * self.vader_weight
            total_weight += self.vader_weight
            
        if 'textblob' in predictions and self.textblob_weight > 0:
            final_scores["negative"] += predictions['textblob']["negative"] * self.textblob_weight
            final_scores["positive"] += predictions['textblob']["positive"] * self.textblob_weight
            total_weight += self.textblob_weight
        
        # Normalize if not all models contributed
        if total_weight > 0 and total_weight != 1.0:
            final_scores["negative"] /= total_weight
            final_scores["positive"] /= total_weight
        
        # Calculate neutral score based on confidence difference
        confidence_diff = abs(final_scores['positive'] - final_scores['negative'])
        final_scores['neutral'] = max(1 - confidence_diff, 0)
        
        return final_scores

    def analyze_text(self, text: str, include_confidence_scores: bool = False) -> Dict:
        """Analyze the sentiment of a given text using ensemble approach."""
        start_time = time()
        
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Get ensemble predictions
        scores = self._get_ensemble_prediction(processed_text)
        
        # Determine final sentiment
        if scores['neutral'] > 0.4:
            sentiment = "neutral"
        elif scores['positive'] > scores['negative']:
            sentiment = "positive"
        else:
            sentiment = "negative"
        
        # Prepare response
        response = {
            "sentiment": sentiment,
            "processing_time": time() - start_time
        }
        
        # Add confidence scores if requested
        if include_confidence_scores:
            response["confidence_scores"] = scores
        
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
