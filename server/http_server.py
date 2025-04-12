from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image
from io import BytesIO
import sys
import os
import time
from typing import List, Dict, Union, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ObjectDetector

app = FastAPI(
    title="Object Detection API",
    description="API for detecting objects in images using Faster R-CNN",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize model
model = ObjectDetector()

# Define Prometheus metrics
INFERENCE_COUNT = Counter('app_http_inference_count_total', 'Number of HTTP endpoint invocations')
PREDICTION_TIME = Histogram('app_prediction_time_seconds', 'Time spent in prediction')
PREDICTION_ERRORS = Counter('app_prediction_errors_total', 'Number of prediction errors')

class PredictRequest(BaseModel):
    url: HttpUrl

class PredictResponse(BaseModel):
    objects: List[str]

class PredictResponseWithConfidence(BaseModel):
    objects: List[Dict[str, Union[str, float]]]

class BatchPredictRequest(BaseModel):
    urls: List[HttpUrl]

class PredictRequestWithOptions(PredictRequest):
    confidence_threshold: float = 0.75
    max_objects: Optional[int] = None

def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/model/info")
async def model_info():
    return {
        "model_name": "Faster R-CNN",
        "version": "1.0",
        "device": model.device.type,
        "categories": model.categories
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        INFERENCE_COUNT.inc()
        with PREDICTION_TIME.time():
            image = download_image(str(request.url))
            objects = model.predict(image)
            return PredictResponse(objects=objects)
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictRequest):
    results = []
    for url in request.urls:
        try:
            image = download_image(str(url))
            objects = model.predict(image)
            results.append({"url": str(url), "objects": objects})
        except Exception as e:
            results.append({"url": str(url), "error": str(e)})
    return {"results": results}

@app.post("/predict_with_confidence", response_model=PredictResponseWithConfidence)
async def predict_with_confidence(request: PredictRequest):
    try:
        image = download_image(str(request.url))
        predictions = model.predict_with_confidence(image)
        return PredictResponseWithConfidence(objects=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_options", response_model=PredictResponse)
async def predict_with_options(request: PredictRequestWithOptions):
    try:
        image = download_image(str(request.url))
        objects = model.predict(image, 
                              confidence_threshold=request.confidence_threshold,
                              max_objects=request.max_objects)
        return PredictResponse(objects=objects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 