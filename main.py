from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from sentiment_analyzer import SentimentAnalyzer
import os
import openai
import logging
from dotenv import load_dotenv
import traceback
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="AI Sentiment Analysis",
    description="AI-Powered Sentiment Analysis Service",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("RENDER_REDIS_HOST", "localhost"),
    port=int(os.getenv("RENDER_REDIS_PORT", 6379)),
    password=os.getenv("RENDER_REDIS_PASSWORD", ""),
    decode_responses=True
)

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for testing
    allow_credentials=False,  # Set to False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_confidence_scores: bool = Field(default=False)

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    include_confidence_scores: bool = Field(default=False)

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """Health check endpoint."""
    return {"status": "ok", "message": "Sentiment Analysis API is running"}

@app.post("/analyze/batch")
@limiter.limit("10/minute")
async def analyze_batch(request: Request, input_data: BatchTextInput):
    """Analyze sentiment of multiple texts in batch."""
    try:
        results = []
        for text in input_data.texts:
            result = await analyzer.analyze_text(
                text,
                input_data.include_confidence_scores
            )
            if "error" in result:
                results.append({
                    "error": result["error"],
                    "sentiment": "neutral",
                    "processing_time": 0.5
                })
            else:
                results.append(result)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch request: {str(e)}"
        )

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_sentiment(request: Request, input_data: TextInput):
    """Analyze sentiment of a single text."""
    try:
        result = await analyzer.analyze_text(
            input_data.text,
            input_data.include_confidence_scores
        )
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
