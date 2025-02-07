import os
from typing import Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the OpenAI-based sentiment analyzer."""
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini-2024-07-18')
        
        # Carefully crafted prompt for accurate sentiment analysis
        self.system_prompt = """You are an expert sentiment analyzer. Your task is to analyze the sentiment of text with high accuracy.
        
        Rules for analysis:
        1. Consider the full context and nuance of the text
        2. Account for sarcasm and implicit meanings
        3. Pay attention to emoticons and emojis
        4. Consider intensity modifiers (very, extremely, etc.)
        5. Identify neutral statements accurately
        
        Output only one of these sentiments: positive, negative, or neutral.
        Also provide confidence scores for each category, ensuring they sum to 1.0."""

    def preprocess_text(self, text: str) -> str:
        """Clean and prepare text for sentiment analysis."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        return text

    async def analyze_text(self, text: str, include_confidence_scores: bool = False) -> Dict:
        """Analyze the sentiment of the given text using OpenAI."""
        try:
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            if not cleaned_text:
                return {"error": "Empty text provided"}

            # Create the user prompt
            user_prompt = f"Analyze the sentiment of this text: '{cleaned_text}'\n\nProvide the sentiment and confidence scores in this exact format:\nSentiment: [sentiment]\nConfidence Scores:\nPositive: [score]\nNegative: [score]\nNeutral: [score]"

            # Get completion from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent results
            )

            # Parse the response
            result = response.choices[0].message.content
            
            # Extract sentiment and confidence scores
            lines = result.strip().split('\n')
            sentiment = lines[0].split(': ')[1].lower()
            confidence_scores = {
                'positive': float(lines[2].split(': ')[1]),
                'negative': float(lines[3].split(': ')[1]),
                'neutral': float(lines[4].split(': ')[1])
            }

            # Prepare response
            response = {
                "sentiment": sentiment,
                "confidence_scores": confidence_scores if include_confidence_scores else None
            }

            return response

        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    def __call__(self, text: str, include_confidence_scores: bool = False) -> Dict:
        """Callable interface for the analyzer."""
        import asyncio
        return asyncio.run(self.analyze_text(text, include_confidence_scores))
