# Object Detection API

A service for object detection in images using Faster R-CNN model. The service provides two interfaces:
- REST API (HTTP) on port 8080
- gRPC API on port 9090

Both interfaces provide the same functionality, including basic object detection, confidence-based detection, and batch processing capabilities.

## Features

- Two API interfaces (REST and gRPC)
- Object detection in images
- Confidence scores for detected objects
- Batch processing of multiple images
- Customizable detection parameters
- Prometheus metrics for monitoring
- Swagger UI for REST API testing

## Quick Start



### Installation
```bash
# Clone the repository
git clone https://github.com/mzabelin8/model_app_deploy.git
cd model_app_deploy

# Install dependencies
pip install -r requirements.txt
```

## REST API

### Running the HTTP Server
```bash
cd server
python http_server.py
```

The REST API will be available at: http://localhost:8080

### REST API Examples

#### 1. Check Server Health
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

#### 2. Get Model Information
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

#### 3. Basic Object Detection
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

#### 4. Detection with Confidence Scores
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

## gRPC API

### Running the gRPC Server
```bash
cd server
python grpc_server.py
```

The gRPC server will be available at: localhost:9090

### Testing gRPC API
We provide a test client that demonstrates all available functionality:
```bash
python grpc_client.py
```

Expected output:
```
Testing HealthCheck:
Status: healthy
Model loaded: True

Testing GetModelInfo:
Model name: Faster R-CNN
Version: 1.0
Device: cpu
Categories: ['__background__', 'person', 'bicycle', 'car', 'motorcycle']...

Testing Predict:
Detected objects: ['dog']

Testing PredictWithConfidence:
Object: dog, Confidence: 0.9965646862983704

Testing BatchPredict:
URL: https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
Objects: ['dog']

Testing PredictWithOptions:
Detected objects: ['dog']
```

### gRPC Service Definition
The gRPC service is defined in `proto/inference.proto`:

```protobuf
service InstanceDetector {
  // Basic prediction
  rpc Predict(PredictRequest) returns (PredictResponse);
  
  // Prediction with confidence scores
  rpc PredictWithConfidence(PredictRequest) returns (PredictWithConfidenceResponse);
  
  // Batch prediction
  rpc BatchPredict(BatchPredictRequest) returns (BatchPredictResponse);
  
  // Prediction with custom options
  rpc PredictWithOptions(PredictWithOptionsRequest) returns (PredictResponse);
  
  // Get model information
  rpc GetModelInfo(Empty) returns (ModelInfo);
  
  // Health check
  rpc HealthCheck(Empty) returns (HealthResponse);
}
```

### Using gRPC API in Python

Here's an example of how to use the gRPC API in your Python code:

```python
import grpc
from proto import inference_pb2
from proto import inference_pb2_grpc

# Create a channel
channel = grpc.insecure_channel('localhost:9090')
stub = inference_pb2_grpc.InstanceDetectorStub(channel)

# Health check
health_response = stub.HealthCheck(inference_pb2.Empty())
print(f"Status: {health_response.status}")

# Basic prediction
predict_response = stub.Predict(
    inference_pb2.PredictRequest(
        url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
)
print(f"Detected objects: {predict_response.objects}")

# Prediction with confidence
confidence_response = stub.PredictWithConfidence(
    inference_pb2.PredictRequest(
        url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
)
for obj in confidence_response.objects:
    print(f"Object: {obj.label}, Confidence: {obj.confidence}")
```

## API Documentation

### REST API Documentation
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

### gRPC API Documentation
The gRPC API is documented in the `proto/inference.proto` file, which serves as the contract between the client and server.

## Project Structure
```
.
├── model/                 # Model module
│   ├── model.py          # Main model class
│   └── test_model.py     # Model tests
├── proto/                 # gRPC definitions
│   ├── inference.proto   # Service definition
│   └── __init__.py      # Python package file
├── server/               # Server module
│   ├── http_server.py    # REST API server
│   ├── grpc_server.py    # gRPC server
│   └── grpc_client.py    # gRPC test client
└── requirements.txt      # Project dependencies
```

## Development

### Running Tests
```bash
cd model
python test_model.py
```

### Regenerating gRPC Code
If you modify the `inference.proto` file, you need to regenerate the Python code:
```bash
python -m grpc_tools.protoc -Iproto --python_out=proto --grpc_python_out=proto proto/inference.proto
```

