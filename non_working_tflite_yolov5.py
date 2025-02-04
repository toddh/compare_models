#!/usr/bin/python3

# REFACTORED TO MOVE THE PiCamera2 calls into a separate module.

# Copyright (c) 2022 Raspberry Pi Ltd
# Author: Zanz2 <https://github.com/Zanz2>
# SPDX-License-Identifier: BSD-3-Clause

# A TensorFlow Lite example for Picamera2 on Raspberry Pi 5 (OS Bookworm) with an active cooler
#
# Install necessary dependences before starting,
#
# $ sudo apt update
# $ sudo apt install build-essential
# $ sudo apt install libatlas-base-dev
# $ sudo apt install python3-pip

# Use system python or if you prefer not to mess with system python:
# install a version manager (like pyenv) and use 'pyenv virtualenv --system-site-packages ENV_NAME'
# $ pip3 install tflite-runtime
# $ pip3 install opencv-python-headless (if using system python: sudo apt install python3-opencv)
#
# and run from the command line,
#
# python3 yolo_v5_real_time_with_labels.py --model=yolov5s-fp16.tflite --label=coco_labels_yolov5.txt

import cv2
import numpy as np
# import tflite_runtime.interpreter as tflite
import tensorflow as tf


IMG_PATH = '/Users/todd/dev/samples/2/bravo-2024-09-21 22.50.36.00544-c0-n-P-Main_.jpg'
LABELS_PATH = '/Users/todd/Documents/dev/compare_models/coco_labels_yolov5.txt'
MODEL_PATH = '/Users/todd/Documents/dev/compare_models/yolov5s-fp16.tflite'


normalSize = (1920, 1080)
lowresSize = (640, 640)  # Shape Yolov5 s was trained with
# if using other yolov5 flavour then image from stream will be resized accordingly.

rectangles = []


def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def DrawRectangles(img):
    for rect in rectangles:
        xmin, ymin, xmax, ymax = rect[0:4]

        rect_start = xmin, ymin
        rect_end = xmax, ymax
        cv2.rectangle(img, rect_start, rect_end, (0, 255, 0, 0))
        if len(rect) == 5:
            text = rect[4]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (xmin, ymin - 10),
                        font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                


def classFilter(classdata):
    return [c.argmax() for c in classdata]
    # generates a list, loop through all predictions and get the best classification location


def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze(output_data[..., 4:5])  # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:])  # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]  # xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]


def main():
    global rectangles

    labels = ReadLabelFile(LABELS_PATH)

    interpreter = tf.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    while True:
        img = cv2.imread(IMG_PATH)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = False
        if input_details[0]['dtype'] == np.float32:
            floating_model = True

        new_shape = (width, height)  # the shape the model was trained with
        if new_shape != lowresSize:
            img = cv2.resize(img, new_shape)

        input_data = np.expand_dims(img, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        xyxy, classes, scores = YOLOdetect(output_data)
        rectangles = []

        for i in range(len(scores)):
            if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
                xmin = int(max(1, (xyxy[0][i] * normalSize[0])))
                ymin = int(max(1, (xyxy[1][i] * normalSize[1])))
                xmax = int(min(normalSize[0], (xyxy[2][i] * normalSize[0])))
                ymax = int(min(normalSize[1], (xyxy[3][i] * normalSize[1])))

                box = [xmin, ymin, xmax, ymax]
                rectangles.append(box)
                if labels:
                    rectangles[-1].append(labels[classes[i]])

        DrawRectangles(img)
        cv2.imshow(f"Did this work?", img)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()