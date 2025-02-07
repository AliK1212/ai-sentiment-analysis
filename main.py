from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sentiment_analyzer import SentimentAnalyzer
import os
import openai
import logging
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found")

# Test OpenAI API connection
try:
    logger.info("Testing OpenAI API connection...")
    test_response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    if test_response['choices'][0]['message']['content']:
        logger.info("OpenAI API connection successful")
    else:
        logger.error("OpenAI API test failed: No content in response")
        raise ValueError("OpenAI API test failed")
except Exception as e:
    logger.error(f"Failed to connect to OpenAI API: {str(e)}")
    logger.error(traceback.format_exc())
    raise

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing text sentiment using OpenAI",
    version="1.0.0"
)

# Add CORS middleware with environment variable configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600  # Cache preflight requests for 10 minutes
)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_confidence_scores: bool = Field(default=False)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Sentiment Analysis API is running"}

@app.post("/analyze")
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of a single text."""
    try:
        result = await analyzer.analyze_text(
            input_data.text,
            input_data.include_confidence_scores
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "sentiment": "neutral",
            "confidence": {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 100.0
            }
        }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
