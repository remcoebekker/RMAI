from ultralytics import YOLO
import os


class Trainer:

    def __init__(self):
        # Create a new YOLO model from scratch
        self.__face_classification_model = YOLO("yolov8n-cls.pt", verbose=False)
        os.environ["YOLO_VERBOSE"] = "False"

    def train(self, data_path, name):
        print(data_path)
        results_class = self.__face_classification_model.train(data=data_path, epochs=25, name=name)
