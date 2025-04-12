import grpc
import sys
import os

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем сгенерированные файлы
from proto import inference_pb2
from proto import inference_pb2_grpc

def run():
    # Создаем канал для подключения к серверу
    channel = grpc.insecure_channel('localhost:9090')
    stub = inference_pb2_grpc.InstanceDetectorStub(channel)

    # Тестируем HealthCheck
    print("Testing HealthCheck:")
    health_response = stub.HealthCheck(inference_pb2.Empty())
    print(f"Status: {health_response.status}")
    print(f"Model loaded: {health_response.model_loaded}\n")

    # Тестируем GetModelInfo
    print("Testing GetModelInfo:")
    model_info = stub.GetModelInfo(inference_pb2.Empty())
    print(f"Model name: {model_info.model_name}")
    print(f"Version: {model_info.version}")
    print(f"Device: {model_info.device}")
    print(f"Categories: {model_info.categories[:5]}...\n")

    # Тестируем Predict
    print("Testing Predict:")
    predict_response = stub.Predict(
        inference_pb2.PredictRequest(
            url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )
    )
    print(f"Detected objects: {predict_response.objects}\n")

    # Тестируем PredictWithConfidence
    print("Testing PredictWithConfidence:")
    confidence_response = stub.PredictWithConfidence(
        inference_pb2.PredictRequest(
            url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )
    )
    for obj in confidence_response.objects:
        print(f"Object: {obj.label}, Confidence: {obj.confidence}")
    print()

    # Тестируем BatchPredict
    print("Testing BatchPredict:")
    batch_response = stub.BatchPredict(
        inference_pb2.BatchPredictRequest(
            urls=[
                "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
                "https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg"
            ]
        )
    )
    for result in batch_response.results:
        print(f"URL: {result.url}")
        if result.objects:
            print(f"Objects: {result.objects}")
        if result.error:
            print(f"Error: {result.error}")
        print()

    # Тестируем PredictWithOptions
    print("Testing PredictWithOptions:")
    options_response = stub.PredictWithOptions(
        inference_pb2.PredictWithOptionsRequest(
            url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            confidence_threshold=0.8,
            max_objects=3
        )
    )
    print(f"Detected objects: {options_response.objects}")

if __name__ == '__main__':
    run() 