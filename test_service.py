import requests
import json

def test_sentiment_analysis():
    """Test the sentiment analysis service with various types of text."""
    
    base_url = "http://localhost:8003"
    test_cases = [
        {
            "text": "I absolutely love this product! It's amazing and works perfectly.",
            "expected": "positive"
        },
        {
            "text": "This is the worst experience ever. Nothing works right.",
            "expected": "negative"
        },
        {
            "text": "The product is okay, it has some good and bad points.",
            "expected": "neutral"
        },
        {
            "text": "😊 Great service! Would recommend! 👍",
            "expected": "positive"
        }
    ]
    
    print("\nTesting Sentiment Analysis Service...")
    print("=====================================")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return
    
    # Test sentiment analysis
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/analyze",
                json={
                    "text": test_case["text"],
                    "include_confidence_scores": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nTest Case {i}:")
                print(f"Text: {test_case['text']}")
                print(f"Expected: {test_case['expected']}")
                print(f"Got: {result['sentiment']}")
                print("Confidence Scores:", json.dumps(result['confidence_scores'], indent=2))
                print(f"Processing Time: {result['processing_time']:.3f}s")
                
                if result['sentiment'] == test_case['expected']:
                    print("✅ Test passed")
                else:
                    print("❌ Test failed")
            else:
                print(f"❌ Test case {i} failed with status code {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test case {i} failed with error: {str(e)}")

if __name__ == "__main__":
    test_sentiment_analysis()
