import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import requests
from io import BytesIO

class ObjectDetector:
    def __init__(self):
        """
        Initialize the object detector.
        Loads a pre-trained Faster R-CNN model with ResNet50 backbone and FPN.
        """
        # Determine the device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load weights and model
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights)
        
        # Set model to evaluation mode and move to the target device
        self.model.eval()
        self.model.to(self.device)
        
        # Get the list of categories (object classes)
        self.categories = self.weights.meta['categories']

    def predict(self, image: Image.Image, confidence_threshold: float = 0.75) -> list[str]:
        """
        Detect objects in an image.
        
        Args:
            image (Image.Image): PIL Image object to analyze
            confidence_threshold (float): Confidence threshold for filtering predictions
            
        Returns:
            list[str]: List of detected object names
        """
        # Prepare the image
        transform = self.weights.transforms()
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(img_tensor)
            
        # Process predictions
        results = []
        pred = predictions[0]
        scores = pred['scores']
        labels = pred['labels']
        
        # Filter by confidence threshold
        mask = scores > confidence_threshold
        filtered_labels = labels[mask]
        
        # Convert class indices to their names
        return [self.categories[label.item()] for label in filtered_labels] 