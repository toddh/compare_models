import numpy as np

# Useful reference https://blog.teclado.com/python-abc-abstract-base-classes/

class AbstractModel:
    def __init__(self):
        pass

    def name(self):
        pass

    def model_path(self):
        pass
    
    def model_name(self):
        pass

    # TODO: Refactor interpreter into abstract model
    def process_model(self, input_details, output_details):
        pass

    def class_names(self):
        pass

    def class_name(self, idx):
        pass

    def data_type(self):
        pass

    def get_results(self, interpreter, output_details):
        """
        The typical output order for object detection models is [boxes, scores, classes,
        num_detections]".
        https://github.com/ultralytics/yolov5/issues/1981#issuecomment-1891162452

        Args:
            interpreter (_type_): _description_
            output_details (_type_): _description_

        Returns:
            array: detection_boxes - an array of shape (num_boxes, 4) with normalized coordinates
            array: detection_classes - an array  of shape (num_boxes, ) containing class prediction for each box
            array: detection_scores - an array of shape (num_boxes, )
            integer: num_boxes - an integer containing the number of detected boxes      
        """
        detection_boxes = interpreter.get_tensor(output_details[0]["index"])
        detection_boxes = detection_boxes[0, ...]
        detection_classes = interpreter.get_tensor(output_details[1]["index"])
        detection_classes = detection_classes[0, ...]
        detection_scores = interpreter.get_tensor(output_details[2]["index"])
        detection_scores = detection_scores[0, :]

        num_boxes = interpreter.get_tensor(output_details[3]["index"])
        return detection_boxes, detection_scores, detection_classes, int(num_boxes[0])