import yaml
import numpy as np
import abstract_model

class EfficientDet(abstract_model.AbstractModel):
    """https://www.kaggle.com/models/tensorflow/efficientdet/tfLite

    Args:
        abstract_model (_type_): _description_
    """
    def __init__(self):
        super().__init__()

        # self._model_path = "tensorflow/efficientdet/tfLite/lite0-detection-default"
        # self._model_path = "tensorflow/efficientdet/tfLite/lite1-detection-default"
        self._model_path = "tensorflow/efficientdet/tfLite/lite2-detection-default"
        # self._model_path = "tensorflow/efficientdet/tfLite/lite3-detection-default"
        # self._model_path = "tensorflow/efficientdet/tfLite/lite4-detection-default"

        self._model_name = "1.tflite"  # If lite4, it's 2.tflite
        self._class_names = []

        with open("coco_labels.txt", "r") as file:
            self._class_names = [line.strip() for line in file.readlines()]

        pass

    def model_path(self):
        return self._model_path
    
    def model_name(self):
        return self._model_name
    
    def class_names(self):
        return self._class_names
    
    def class_name(self, idx):
        str = self._class_names[idx]
        return str
    
    def data_type(self):
        return np.uint8
    
    def name(self):
        return "Efficient Det"