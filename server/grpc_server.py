import grpc
from concurrent import futures
import time
import sys
import os
import requests
from PIL import Image
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ObjectDetector
from proto import inference_pb2
from proto import inference_pb2_grpc

class InstanceDetectorServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.model = ObjectDetector()

    def download_image(self, url: str) -> Image.Image:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise grpc.RpcError(grpc.StatusCode.INVALID_ARGUMENT, f"Failed to download image: {str(e)}")

    def Predict(self, request, context):
        try:
            image = self.download_image(request.url)
            objects = self.model.predict(image)
            return inference_pb2.PredictResponse(objects=objects)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()

    def PredictWithConfidence(self, request, context):
        try:
            image = self.download_image(request.url)
            predictions = self.model.predict_with_confidence(image)
            objects = [
                inference_pb2.ObjectWithConfidence(label=pred["label"], confidence=pred["confidence"])
                for pred in predictions
            ]
            return inference_pb2.PredictWithConfidenceResponse(objects=objects)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictWithConfidenceResponse()

    def BatchPredict(self, request, context):
        results = []
        for url in request.urls:
            try:
                image = self.download_image(url)
                objects = self.model.predict(image)
                results.append(
                    inference_pb2.BatchPredictResult(url=url, objects=objects)
                )
            except Exception as e:
                results.append(
                    inference_pb2.BatchPredictResult(url=url, error=str(e))
                )
        return inference_pb2.BatchPredictResponse(results=results)

    def PredictWithOptions(self, request, context):
        try:
            image = self.download_image(request.url)
            objects = self.model.predict(
                image,
                confidence_threshold=request.confidence_threshold,
                max_objects=request.max_objects
            )
            return inference_pb2.PredictResponse(objects=objects)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()

    def GetModelInfo(self, request, context):
        return inference_pb2.ModelInfo(
            model_name="Faster R-CNN",
            version="1.0",
            device=self.model.device.type,
            categories=self.model.categories
        )

    def HealthCheck(self, request, context):
        return inference_pb2.HealthResponse(
            status="healthy",
            model_loaded=True
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(
        InstanceDetectorServicer(), server
    )
    server.add_insecure_port('[::]:9090')
    server.start()
    print("gRPC server started on port 9090")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve() 