# Sentiment Analysis API

A powerful sentiment analysis API built with FastAPI and Hugging Face's DistilBERT model. This API provides real-time sentiment analysis of text with confidence scores and processing time metrics.

## Features

- **Real-time Sentiment Analysis**: Analyze text sentiment using state-of-the-art NLP models
- **Confidence Scores**: Get detailed confidence scores for positive, negative, and neutral sentiments
- **Batch Processing**: Analyze multiple texts in a single request
- **Processing Time Metrics**: Track analysis performance with processing time information
- **CORS Support**: Built-in CORS middleware for cross-origin requests
- **Modern UI**: Beautiful React-based interface with interactive features

## Technical Stack

- **Backend**:
  - FastAPI: High-performance web framework
  - Hugging Face Transformers: State-of-the-art NLP models
  - DistilBERT: Lightweight, efficient BERT model
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
    "positive": 0.92,
    "negative": 0.05,
    "neutral": 0.03
  },
  "processing_time": 0.156
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
        "positive": 0.92,
        "negative": 0.05,
        "neutral": 0.03
      }
    },
    {
      "sentiment": "negative",
      "confidence_scores": {
        "positive": 0.15,
        "negative": 0.80,
        "neutral": 0.05
      }
    }
  ],
  "processing_time": 0.312
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_NAME | Hugging Face model identifier | "distilbert-base-uncased-finetuned-sst-2-english" |
| CACHE_DIR | Directory for model caching | "./model_cache" |
| MAX_LENGTH | Maximum text length | 512 |
| BATCH_SIZE | Batch processing size | 32 |
| PORT | Server port | 8000 |
| HOST | Server host | "0.0.0.0" |

## Getting Started

1. **Clone the Repository**:
```bash
git clone <repository-url>
cd sentiment-analysis-api
```

2. **Set Up Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure Environment Variables**:
Create a `.env` file with your configuration:
```env
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
CACHE_DIR=./model_cache
MAX_LENGTH=512
BATCH_SIZE=32
PORT=8000
HOST=0.0.0.0
```

4. **Run with Docker**:
```bash
docker-compose up sentiment-analysis-api
```

Or run locally:
```bash
uvicorn main:app --reload --port 8000
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input text or parameters
- **422 Validation Error**: Request body validation failures
- **500 Internal Server Error**: Model loading or processing errors

Example error response:
```json
{
  "detail": "Text exceeds maximum length of 512 characters"
}
```

## Performance Considerations

- The model is loaded once at startup and cached
- Batch processing is more efficient for multiple texts
- Text length affects processing time
- First request may be slower due to model loading

## UI Features

The included React UI provides:
- Interactive text input with single and batch modes
- Real-time sentiment analysis
- Confidence score visualization
- Example cases with explanations
- Batch processing interface with:
  - Dynamic text input fields
  - Add/remove text entries
  - Batch example cases
  - Individual results for each text
- Tooltips for understanding scores
- Copy-to-clipboard functionality
- Responsive design
- Loading states and error handling

## Batch Processing

The API and UI support efficient batch processing of multiple texts:

### Benefits
- **Efficiency**: Process multiple texts in a single API call
- **Consistency**: Get uniform analysis across all texts
- **Performance**: Reduced overhead compared to multiple single requests

### Batch Size Limits
- Maximum texts per batch: 100
- Maximum text length: 512 characters per text
- Optimal batch size: 32 texts (configurable)

### Example Use Cases
1. **Customer Feedback Analysis**:
   ```json
   {
     "texts": [
       "The customer service was excellent!",
       "Product quality needs improvement",
       "Shipping was fast and reliable"
     ]
   }
   ```

2. **Social Media Monitoring**:
   ```json
   {
     "texts": [
       "Love the new features! #greatapp",
       "App keeps crashing after update",
       "Nice interface design"
     ]
   }
   ```

3. **Survey Response Analysis**:
   ```json
   {
     "texts": [
       "Very satisfied with the service",
       "Could be better but works fine",
       "Not what I expected"
     ]
   }
   ```

### Best Practices
1. **Optimal Batch Size**: Use the default batch size (32) for best performance
2. **Error Handling**: Implement proper error handling for partial batch failures
3. **Rate Limiting**: Consider rate limiting for large batch requests
4. **Monitoring**: Track batch processing times for optimization
