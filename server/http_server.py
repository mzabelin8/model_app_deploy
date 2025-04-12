from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image
from io import BytesIO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ObjectDetector
import uvicorn
from typing import List
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI()
model = ObjectDetector()

# Define Prometheus metrics
INFERENCE_COUNT = Counter('app_http_inference_count_total', 'Number of HTTP endpoint invocations')

class PredictRequest(BaseModel):
    url: HttpUrl

class PredictResponse(BaseModel):
    objects: List[str]

def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Increment the inference counter
        INFERENCE_COUNT.inc()
        
        # Download image from URL
        image = download_image(str(request.url))
        
        # Get predictions from model
        objects = model.predict(image)
        
        return PredictResponse(objects=objects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 