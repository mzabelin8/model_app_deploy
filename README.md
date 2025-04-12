# Object Detection API

A REST API service for object detection in images using Faster R-CNN model. The service provides various endpoints for image processing, including basic object detection, confidence-based detection, and batch processing capabilities.

## Quick Start

### Requirements
- Python 3.9 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/mzabelin8/model_app_deploy.git
cd model_app_deploy

# Install dependencies
pip install -r requirements.txt
```

### Running the Server
```bash
cd server
python http_server.py
```

The server will be available at: http://localhost:8080

## Examples

### 1. Check Server Health
```bash
curl http://localhost:8080/health
```
Expected response:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```
This endpoint verifies that the server is running and the model is loaded correctly.

### 2. Get Model Information
```bash
curl http://localhost:8080/model/info
```
Expected response:
```json
{
    "model_name": "Faster R-CNN",
    "version": "1.0",
    "device": "cpu",
    "categories": ["person", "car", "dog", ...]
}
```
This endpoint provides information about the loaded model, including supported object categories.

### 3. Basic Object Detection
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"}'
```
Expected response:
```json
{
    "objects": ["dog"]
}
```
This endpoint detects objects in the image and returns their labels.

### 4. Detection with Confidence Scores
```bash
curl -X POST "http://localhost:8080/predict_with_confidence" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"}'
```
Expected response:
```json
{
    "objects": [
        {
            "label": "dog",
            "confidence": 0.996
        }
    ]
}
```
This endpoint provides both object labels and confidence scores for each detection.

### 5. Batch Processing
```bash
curl -X POST "http://localhost:8080/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{
         "urls": [
             "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
             "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg"
         ]
     }'
```
Expected response:
```json
{
    "results": [
        {
            "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            "objects": ["dog"]
        },
        {
            "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg",
            "objects": ["cat"]
        }
    ]
}
```
This endpoint processes multiple images in a single request.

### 6. Detection with Custom Parameters
```bash
curl -X POST "http://localhost:8080/predict_with_options" \
     -H "Content-Type: application/json" \
     -d '{
         "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
         "confidence_threshold": 0.8,
         "max_objects": 3
     }'
```
Expected response:
```json
{
    "objects": ["dog"]
}
```
This endpoint allows you to customize detection parameters like confidence threshold and maximum number of objects.

## Features

- **Object Detection**: Detect objects in images using Faster R-CNN model
- **Confidence Scores**: Get confidence scores for each detected object
- **Batch Processing**: Process multiple images in a single request
- **Customizable Parameters**: Adjust confidence threshold and maximum number of objects
- **REST API**: Easy-to-use HTTP endpoints
- **Swagger UI**: Interactive API documentation at `/docs`
- **Prometheus Metrics**: Monitor API performance and usage
- **Health Checks**: Verify server and model status
- **Model Information**: Get details about the loaded model
- **Error Handling**: Comprehensive error responses
