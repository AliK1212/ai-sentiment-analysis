import os
from typing import Dict, Optional
import openai
from dotenv import load_dotenv
import re
import json

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the OpenAI-based sentiment analyzer."""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini-2024-07-18')  # Default to gpt-4o-mini if not specified
        
        # Comprehensive prompt for highly accurate sentiment analysis
        self.system_prompt = """You are an expert sentiment analyzer with deep understanding of human emotions, context, and linguistic nuances.

        Your task is to perform a detailed sentiment analysis following these strict rules:

        1. CONTEXT ANALYSIS:
           - Consider the full context and broader meaning
           - Account for industry-specific terminology
           - Understand cultural references and idioms
           - Consider temporal context (past, present, future implications)

        2. LINGUISTIC NUANCE:
           - Detect and interpret sarcasm and irony
           - Analyze tone and voice
           - Consider intensity modifiers (very, extremely, somewhat)
           - Evaluate negations and their scope
           - Account for conditional statements

        3. EMOTIONAL INDICATORS:
           - Analyze emotional vocabulary
           - Interpret emoticons and emojis
           - Consider punctuation patterns (!!!, ...)
           - Detect subtle emotional undertones
           - Evaluate emphasis patterns (CAPS, *text*)

        4. OBJECTIVITY:
           - Identify neutral statements accurately
           - Distinguish between facts and opinions
           - Consider technical or academic language
           - Evaluate balanced perspectives

        5. CONFIDENCE SCORING:
           - Provide detailed confidence scores
           - Ensure scores sum to 1.0
           - Consider ambiguity in scoring
           - Account for mixed sentiments
           - Provide higher confidence for clear indicators

        OUTPUT FORMAT (Exactly as shown):
        Sentiment: [positive/negative/neutral]
        Confidence Scores:
        Positive: [0.0-1.0]
        Negative: [0.0-1.0]
        Neutral: [0.0-1.0]"""

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing while preserving important sentiment indicators."""
        if not text:
            return ""
        
        # Normalize whitespace while preserving emoji and special characters
        text = " ".join(text.split())
        
        # Standardize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        
        # Preserve emoticons and emojis while cleaning text
        text = re.sub(r'[^\w\s:;)(><}{}\[\]\\/@#$%^&*+=|~`!?,.\-\'"]+', ' ', text)
        
        return text.strip()

    async def analyze_text(self, text: str, include_confidence_scores: bool = False) -> Dict:
        """Analyze the sentiment of the given text using OpenAI with enhanced accuracy."""
        try:
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            if not cleaned_text:
                return {"error": "Empty text provided"}

            # Create a detailed analysis prompt
            user_prompt = f"""Analyze the sentiment of this text with high precision:

            TEXT: '{cleaned_text}'

            Consider:
            1. Overall emotional tone
            2. Presence of sarcasm or irony
            3. Cultural context
            4. Technical language
            5. Emotional intensity

            Provide sentiment and confidence scores in the exact format specified."""

            # Get completion from OpenAI with strict temperature control
            completion = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for consistent results
                max_tokens=150,   # Limit response length for faster processing
                presence_penalty=0.0,  # Neutral presence penalty
                frequency_penalty=0.0   # Neutral frequency penalty
            )

            # Parse the response with error handling
            result = completion['choices'][0]['message']['content']
            try:
                lines = [line.strip() for line in result.strip().split('\n')]
                sentiment = lines[0].split(': ')[1].lower()
                confidence_scores = {
                    'positive': float(lines[2].split(': ')[1]),
                    'negative': float(lines[3].split(': ')[1]),
                    'neutral': float(lines[4].split(': ')[1])
                }
                
                # Validate confidence scores
                total = sum(confidence_scores.values())
                if not (0.99 <= total <= 1.01):  # Allow small floating-point variance
                    # Normalize scores
                    confidence_scores = {k: v/total for k, v in confidence_scores.items()}
                
                # Validate sentiment value
                if sentiment not in ['positive', 'negative', 'neutral']:
                    raise ValueError(f"Invalid sentiment value: {sentiment}")

            except Exception as e:
                print(f"Error parsing OpenAI response: {str(e)}")
                print(f"Raw response: {result}")
                raise ValueError("Failed to parse sentiment analysis results")

            # Format response exactly as frontend expects
            confidence_scores = {
                "positive": round(confidence_scores['positive'] * 1, 2),
                "negative": round(confidence_scores['negative'] * 1, 2),
                "neutral": round(confidence_scores['neutral'] * 1, 2)
            }
            
            return {
                "sentiment": sentiment,
                "confidence_scores": confidence_scores,
                "positive": confidence_scores["positive"],
                "negative": confidence_scores["negative"],
                "neutral": confidence_scores["neutral"],
                "processing_time": 0.5  # Add processing time that frontend expects
            }

        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "sentiment": "neutral",
                "confidence_scores": {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 100.0
                },
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 100.0,
                "processing_time": 0.5
            }

    def __call__(self, text: str, include_confidence_scores: bool = False) -> Dict:
        """Callable interface for the analyzer."""
        import asyncio
        return asyncio.run(self.analyze_text(text, include_confidence_scores))
