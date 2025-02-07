# AI Sentiment Analysis

A sophisticated sentiment analysis service that uses an ensemble of state-of-the-art models to provide highly accurate sentiment analysis. Built with FastAPI, this service combines multiple NLP approaches including transformer models, rule-based analysis, and traditional NLP techniques.

## Key Features

- **Ensemble Model Approach**: Combines multiple models for superior accuracy:
  - DistilBERT: Fast and efficient transformer model
  - RoBERTa: Advanced transformer model optimized for social media
  - VADER: Rule-based sentiment analysis
  - TextBlob: Traditional NLP approach
- **Smart Text Preprocessing**: Enhanced text cleaning while preserving meaningful elements
- **Weighted Predictions**: Intelligent combination of multiple model predictions
- **Accurate Neutral Detection**: Improved detection of neutral sentiment
- **Batch Processing**: Analyze multiple texts efficiently
- **Processing Time Metrics**: Track performance with detailed timing information
- **CORS Support**: Built-in CORS middleware for cross-origin requests
- **Modern UI**: Beautiful React-based interface with interactive features

## Technical Stack

- **Backend**:
  - FastAPI: High-performance web framework
  - Multiple Transformer Models: DistilBERT and RoBERTa
  - VADER Sentiment: Rule-based analysis
  - TextBlob: Natural language processing toolkit
  - Pydantic: Data validation
  - Uvicorn: ASGI server

- **Frontend**:
  - React with TypeScript
  - Framer Motion: Smooth animations
  - Tailwind CSS: Utility-first styling
  - Lucide Icons: Modern icon set

## API Endpoints

### 1. Analyze Single Text
```http
POST /analyze
```

**Request Body**:
```json
{
  "text": "Your text to analyze",
  "include_confidence_scores": true
}
```

**Response**:
```json
{
  "sentiment": "positive",
  "confidence_scores": {
    "positive": 0.85,
    "negative": 0.10,
    "neutral": 0.05
  },
  "processing_time": 0.256
}
```

### 2. Batch Analysis
```http
POST /analyze/batch
```

**Request Body**:
```json
{
  "texts": ["Text 1", "Text 2"],
  "include_confidence_scores": true
}
```

**Response**:
```json
{
  "results": [
    {
      "sentiment": "positive",
      "confidence_scores": {
        "positive": 0.85,
        "negative": 0.10,
        "neutral": 0.05
      }
    },
    {
      "sentiment": "negative",
      "confidence_scores": {
        "positive": 0.15,
        "negative": 0.75,
        "neutral": 0.10
      }
    }
  ],
  "processing_time": 0.512
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HOST | Server host address | "0.0.0.0" |
| PORT | Server port number | 8000 |
| CACHE_DIR | Directory for model caching | ".cache" |
| MAX_LENGTH | Maximum text length | 512 |
| BATCH_SIZE | Batch size for processing | 32 |
| DISTILBERT_MODEL | DistilBERT model identifier | "distilbert-base-uncased-finetuned-sst-2-english" |
| ROBERTA_MODEL | RoBERTa model identifier | "cardiffnlp/twitter-roberta-base-sentiment-latest" |
| DISTILBERT_WEIGHT | Weight for DistilBERT predictions | 0.3 |
| ROBERTA_WEIGHT | Weight for RoBERTa predictions | 0.3 |
| VADER_WEIGHT | Weight for VADER predictions | 0.2 |
| TEXTBLOB_WEIGHT | Weight for TextBlob predictions | 0.2 |
| ALLOWED_ORIGINS | Comma-separated list of allowed CORS origins | "http://localhost:3000,http://localhost:8000" |

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
uvicorn main:app --reload
```

## Model Weights

The ensemble approach uses the following weights for combining predictions:
- DistilBERT: 30%
- RoBERTa: 30%
- VADER: 20%
- TextBlob: 20%

This combination provides optimal accuracy across different types of text:
- Formal writing
- Social media content
- Customer reviews
- News articles
- Casual conversations

## Example Use Cases

1. **Customer Feedback Analysis**:
```json
{
  "text": "The new feature is amazing! Much better than before.",
  "include_confidence_scores": true
}
```

2. **Social Media Monitoring**:
```json
{
  "texts": [
    "Love the new features! #greatapp 😍",
    "This update is terrible, nothing works anymore 😡",
    "Interesting changes, need time to get used to them 🤔"
  ]
}
```

3. **Business Reviews**:
```json
{
  "text": "Great service but slightly expensive. Staff was very friendly.",
  "include_confidence_scores": true
}
```

## Performance

The ensemble approach significantly improves accuracy compared to single-model solutions:
- Better handling of mixed sentiments
- More accurate neutral sentiment detection
- Improved understanding of context and nuance
- Enhanced handling of informal text and emoticons
