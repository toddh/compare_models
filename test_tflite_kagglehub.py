#!/usr/bin/python3

"""
Tests Tensorflo Lite models that are pulled from kagglehub.  The workings include:

- EfficientDet
- MobileObject Localizer
- Maybe finally, YOLOv5s

"""

# The following line gets over the fact that setuptools isn't part of python 3.12. Tflite seems to require this.
# See https://stackoverflow.com/a/78136410
import setuptools.dist

import argparse

import numpy as np
import tensorflow as tf
import kagglehub
import os
import cv2
from efficient_det import EfficientDet
from mobile_object_localizer import MobileObjectLocalizer
from kh_yolo5 import YOLOv5

from pprint import pprint

# IMG_DIR = "/Users/todd/dev/samples/single"
IMG_DIR = "/Users/todd/dev/samples/2"
# IMG_DIR = "/Users/todd/dev/images/three"
# IMG_DIR = "/Users/todd/dev/minicoco/data/images/"

option_preview = True

class RunModel:
    def __init__(self, model):
        self.__model = model
        path = kagglehub.model_download(self.__model.model_path())

        file_name = path + "/" + self.__model.model_name()

        with open(file_name, "rb") as f:
            model_content = f.read()
        self._interpreter = tf.lite.Interpreter(model_content=model_content)

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_shape = self._input_details[0]["shape"]
        self._class_names = self.__model.class_names()
        self.__model.process_model(self._input_details, self._output_details)

    def preprocess_image(self, img, input_shape):
        input_data = cv2.resize(img, (input_shape[1], input_shape[2]))
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)  # TODO: Figure out if I'm always doing this right
        input_data = np.clip(input_data, 0, 255)
        input_data = input_data.astype(self.__model.data_type())
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape
        input_data = self.preprocess_image(img, self._input_shape)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        # The TFLite_Detection_PostProcess custom op node has four outputs. THIS IS ONLY TRUE FOR SOME MODELS.
        #
        # detection_boxes: a tensor of shape [1, num_boxes, 4] with normalized coordinates
        # detection_scores: a tensor of shape [1, num_boxes]
        # detection_classes: a tensor of shape [1, num_boxes] containing class prediction for each box
        # num_boxes: a tensor of size 1 containing the number of detected boxes
        # From https://github.com/tensorflow/tensorflow/issues/34761

        datum = []

        detection_boxes, detection_scores, detection_classes, num_boxes = self.__model.get_results(self._interpreter, self._output_details)
        #
        # Using efficient_det to double check outputs, here's what I get:
        # detection_boxes is an array (NOT A TENSOR) shape (25, 4) - values are nomarmalized between 0 and 1
        #   They should be in the form of [xmin, ymin, xmax, ymax]
        # detection_scores is an array (NOT A TENSOR) shape (25, ) - scores are between 0 and 1
        # detection_classes is an array (NOT A TENSOR) shape (25, ) - values are 17.0, 17.0, 32,0, 0.0 etc.....
        # num_boxes is an integer = 25
        # (I think the 25 is the number of output nodes in the neural network.)
        #
        # Mobilenet_object detection is similar
        # detection_boxes shape (100, 4)
        # detection_classes = (100,) - I think mmobile_net_object_detectiojn this will always be zero - it doesn't identify
        # detection_soces is shape (100,)
        # num_boxes is an integer = 100

        for i in range(0, num_boxes):
            if detection_scores[i] > 0.2:
                # pprint(detection_boxes)
                x = detection_boxes[i, [1, 3]] * img_width
                # efficient_det: detection_boxes shape = (25,4)
                # kh_yolo5: an array of arrays.
                # x = [1, 1]
                # x[0] = detection_boxes[i, 1] * img_width
                # x[1] = detection_boxes[i, 3] * img_width

                y = detection_boxes[i, [0, 2]] * img_height
                # y = [1, 1]
                # y[0] = detection_boxes[i, 0] * img_width
                # y[1] = detection_boxes[i, 2] * img_width

                rectangle = [x[0], y[0], x[1], y[1]]
                class_id = detection_classes[i]
                # print(f"{i}: Class: {class_id} score: {detection_scores[0, i]} box: ({str(rectangle)})")
                datum.append(
                    {"class_id": class_id, "score": float(detection_scores[i]), "rectangle": str(rectangle)}
                )
                if option_preview:
                    cv2.rectangle(img, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), 2)
                    class_name = self.__model.class_name(int(class_id))
                    cv2.putText(
                        img,
                        f"{int(class_id)}: {class_name} ({detection_scores[i]:.2f})",
                        (int(x[0]), int(y[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )


        if option_preview and len(datum) > 0:
            cv2.imshow(f"Found {len(datum)} classes", img)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()

        return datum

    def process_directory(self):
        stats = []

        for filename in os.listdir(IMG_DIR):
            img_path = os.path.join(IMG_DIR, filename)
            datum = self.process_image(img_path)

            stats.append(datum)

        cnt = 0
        for datum in stats:
            if len(datum) > 0:
                cnt += 1

        # for datum in stats:
        #     print(str(datum))

        print(f"{self.__model.name()}: num_files {len(stats)} with_objects {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    prog="Process", description="Runs model on images."
)

    parser.add_argument("-p", "--preview", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()
    if args.preview:
        option_preview = True

    # runner = RunModel(MobileObjectLocalizer()).process_directory()
    runner = RunModel(EfficientDet()).process_directory()
    # runner = RunModel(YOLOv5()).process_directory()

    print("Completed.")



