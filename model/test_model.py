import requests
from PIL import Image
from io import BytesIO
from model import ObjectDetector

def test_model():
    # Initialize the detector
    detector = ObjectDetector()
    
    # Test image URL
    image_url = "http://images.cocodataset.org/val2017/000000001268.jpg"
    
    # Download and open the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Get predictions
    objects = detector.predict(image)
    
    # Print results
    print("Detected objects:", objects)

if __name__ == "__main__":
    test_model() 