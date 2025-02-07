from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sentiment_analyzer import SentimentAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing text sentiment using ensemble of models",
    version="1.0.0"
)

# Add CORS middleware with environment variable configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_confidence_scores: bool = Field(default=False)

class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    include_confidence_scores: bool = Field(default=False)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Sentiment Analysis API is running"}

@app.post("/analyze")
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of a single text."""
    try:
        result = analyzer.analyze_text(
            input_data.text,
            input_data.include_confidence_scores
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze/batch")
async def analyze_batch(input_data: BatchInput):
    """Analyze sentiment of multiple texts."""
    try:
        results = analyzer.analyze_batch(
            input_data.texts,
            input_data.include_confidence_scores
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
