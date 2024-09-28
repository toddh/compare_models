import yaml
import numpy as np
from abstract_model import AbstractModel

# https://www.kaggle.com/models/google/mobile-object-localizer-v1
# Input
#
# Inputs are expected to be 3-channel RGB color images of size 192 x 192.
#
# Output
#
# This model outputs four tensors:
#
# num_detections: Total number of detections.
# detection_boxes: Bounding box for each detection.
# detection_scores: Confidence scores for each detection.
# detection_classes: Object class for each detection. Note that this model supports only one class. Labelmap.

# for input in input_details:
#     print(f"Input index: {input['index']}")
#     print(f"Input shape: {input['shape']}")
#     print(f"Input type: {input['dtype']}")
#     print("---")

# mobile_object_localizer_v1 says
# Input index: 0
# Input shape: [  1 320 320   3]
# Input type: <class 'numpy.uint8'>
#
# What does this mean?
# The input_details array has one element. The element has these attributes:
# - index: Refers to a specific tensor in the model. 
# - shape: The required "shape" of the input tensor. [batch_size, heigh, width, channels]. Batch-size has to do with training
# - dtype: The data type
# See https://www.perplexity.ai/search/explain-the-index-attribute-fr-iKh.eS_bSwWhEVir12g_vw


# for output in output_details:
#     print(f"Output index: {output['index']}")
#     print(f"Output shape: {output['shape']}")
#     print(f"Output type: {output['dtype']}")
#     print("---")


class MobileObjectLocalizer(AbstractModel):
    def __init__(self):
        super().__init__()

        self._model_path = "google/mobile-object-localizer-v1/tfLite/default"
        self._model_name = "1.tflite"
        self._class_names = []

        with open("/Users/todd/Documents/dev/yolov5/data/coco.yaml", "r") as file:
            coco_data = yaml.safe_load(file)
            self._class_names = coco_data["names"]

    def model_path(self):
        return self._model_path
    
    def model_name(self):
        return self._model_name
    
    def class_names(self):
        return self._class_names
    
    def data_type(self):
        return np.uint8
    
    def name(self):
        return "Mobile Object Localizer"