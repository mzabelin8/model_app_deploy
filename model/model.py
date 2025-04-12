import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict, Union, Optional

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

    def predict(self, image: Image.Image, confidence_threshold: float = 0.75, max_objects: Optional[int] = None) -> List[str]:
        """
        Detect objects in an image.
        
        Args:
            image (Image.Image): PIL Image object to analyze
            confidence_threshold (float): Confidence threshold for filtering predictions
            max_objects (Optional[int]): Maximum number of objects to return
            
        Returns:
            List[str]: List of detected object names
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
        filtered_scores = scores[mask]
        
        # Sort by confidence score
        sorted_indices = torch.argsort(filtered_scores, descending=True)
        filtered_labels = filtered_labels[sorted_indices]
        
        # Apply max_objects limit if specified
        if max_objects is not None:
            filtered_labels = filtered_labels[:max_objects]
        
        # Convert class indices to their names
        return [self.categories[label.item()] for label in filtered_labels]

    def predict_with_confidence(self, image: Image.Image, confidence_threshold: float = 0.75) -> List[Dict[str, Union[str, float]]]:
        """
        Detect objects in an image with confidence scores.
        
        Args:
            image (Image.Image): PIL Image object to analyze
            confidence_threshold (float): Confidence threshold for filtering predictions
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of dictionaries containing object names and confidence scores
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
        filtered_scores = scores[mask]
        
        # Sort by confidence score
        sorted_indices = torch.argsort(filtered_scores, descending=True)
        filtered_labels = filtered_labels[sorted_indices]
        filtered_scores = filtered_scores[sorted_indices]
        
        # Convert to list of dictionaries
        for label, score in zip(filtered_labels, filtered_scores):
            results.append({
                "label": self.categories[label.item()],
                "confidence": score.item()
            })
            
        return results 