import requests
from PIL import Image
from io import BytesIO
from model import ObjectDetector
import time

def load_test_image(url: str) -> Image.Image:
    """Helper function to load test image"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def test_basic_prediction():
    """Test basic object detection"""
    print("\n=== Testing basic prediction ===")
    detector = ObjectDetector()
    image = load_test_image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg")
    objects = detector.predict(image)
    print("Detected objects:", objects)
    assert len(objects) > 0, "Should detect at least one object"

def test_prediction_with_confidence():
    """Test object detection with confidence scores"""
    print("\n=== Testing prediction with confidence ===")
    detector = ObjectDetector()
    image = load_test_image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg")
    predictions = detector.predict_with_confidence(image)
    print("Predictions with confidence:")
    for pred in predictions:
        print(f"- {pred['label']}: {pred['confidence']:.2f}")
    assert len(predictions) > 0, "Should detect at least one object"
    assert all('confidence' in pred for pred in predictions), "Each prediction should have a confidence score"

def test_prediction_with_options():
    """Test object detection with custom options"""
    print("\n=== Testing prediction with options ===")
    detector = ObjectDetector()
    image = load_test_image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg")
    
    # Test with high confidence threshold
    print("\nTesting with high confidence threshold (0.9):")
    high_conf_objects = detector.predict(image, confidence_threshold=0.9)
    print("High confidence objects:", high_conf_objects)
    
    # Test with low confidence threshold
    print("\nTesting with low confidence threshold (0.3):")
    low_conf_objects = detector.predict(image, confidence_threshold=0.3)
    print("Low confidence objects:", low_conf_objects)
    
    # Test with max objects limit
    print("\nTesting with max objects limit (2):")
    limited_objects = detector.predict(image, max_objects=2)
    print("Limited objects:", limited_objects)
    assert len(limited_objects) <= 2, "Should not return more than 2 objects"

def test_model_performance():
    """Test model performance and timing"""
    print("\n=== Testing model performance ===")
    detector = ObjectDetector()
    image = load_test_image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg")
    
    # Measure prediction time
    start_time = time.time()
    objects = detector.predict(image)
    end_time = time.time()
    
    print(f"Prediction time: {(end_time - start_time):.2f} seconds")
    print(f"Number of objects detected: {len(objects)}")
    print(f"Running on device: {detector.device}")

def run_all_tests():
    """Run all test functions"""
    test_basic_prediction()
    test_prediction_with_confidence()
    test_prediction_with_options()
    test_model_performance()
    print("\n=== All tests completed successfully ===")

if __name__ == "__main__":
    run_all_tests() 