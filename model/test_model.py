from model import ObjectDetector

def main():
    # Create detector instance
    detector = ObjectDetector()
    
    # Test image from COCO dataset
    image_url = "http://images.cocodataset.org/val2017/000000001268.jpg"
    
    # Get predictions
    objects = detector.predict(image_url)
    
    # Print results
    print("Detected objects:")
    for obj in objects:
        print(f"- {obj}")

if __name__ == "__main__":
    main() 