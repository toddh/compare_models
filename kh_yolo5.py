import yaml
import numpy as np
import abstract_model
import tensorflow as tf
from pprint import pprint

# https://www.kaggle.com/models/kaggle/yolo-v5
# About Yolo-v5
# YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset and significantly outperform its previous state-of-the-art yolo models.
#
# This model is trained on the COCO dataset.
#
# The TFLite model takes images as input and detects objects, makes bounding boxes and gives that image as output..
#
# Pre-trained Model weights are provided in this repository
#
# Model is quantized using dynamic range and float16 quantization method as described here.
#
# You can use this to convert the pre-trained models to TFLite Format.
#
# You can use Inference Notebook with TFLite Yolo-v5 model for object detection.
#
# References
#
# TensorFlow Lite Conversion https://ai.google.dev/edge/litert/models/convert
# Float16 quantization in TensorFlow Lite https://ai.google.dev/edge/litert/models/post_training_float16_quant
# Dynamic-range quantization in TensorFlow Lite https://www.tensorflow.org/lite/performance/post_training_quant
#
# See this for return values:
#
# return x[0]  # output only first tensor [1,6300,85] = [xywh, conf, class0, class1, ...]
# this was commented out - but explains what's in that tensor
#       x = x[0][0]  # [x(1,6300,85), ...] to x(6300,85)
#       xywh = x[..., :4]  # x(6300,4) boxes
#       conf = x[..., 4:5]  # x(6300,1) confidences
#       cls = tf.reshape(tf.cast(tf.argmax(x[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
#       return tf.concat([conf, cls, xywh], 1)
# From https://github.com/zldrobit/yolov5/blob/c761637b51701565a9efbd585a05f093f8aa5f41/models/tf.py#L328


class YOLOv5(abstract_model.AbstractModel):
    def __init__(self):
        super().__init__()

        self._model_path = "kaggle/yolo-v5/tfLite/tflite-tflite-model"
        self._model_name = "1.tflite"
        self._class_names = []

        with open("coco.yaml", "r") as file:
            coco_data = yaml.safe_load(file)
            self._class_names = coco_data["names"]

    def model_path(self):
        return self._model_path

    def model_name(self):
        return self._model_name

    def class_names(self):
        return self._class_names
    
    def class_name(self, idx):
        return self._class_names[idx] if idx < len(self._class_names) else "Error"

    def data_type(self):
        return np.float32

    def name(self):
        return "YOLOV5"


    def process_model(self, input_details, output_details):
        print("YOLOV5")

        # print()
        # print("Input details:")
        # for detail in input_details:
        #     pprint(detail)

        # print()
        # print("Output details:")
        # for detail in output_details:
        #     pprint(detail)

    def __compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def __nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.__compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # Update the indices
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def get_results(self, interpreter, output_details):
        """_summary_

        Args:
            interpreter (_type_): _description_
            output_details (_type_): _description_

        Returns:
            array: detection_boxes - an array of shape (num_boxes, 4) with normalized coordinates
            array: detection_classes - an array  of shape (num_boxes, ) containing class prediction for each box
            array: detection_scores - an array of shape (num_boxes, )
            integer: num_boxes - an integer containing the number of detected boxes

        """

        # For YOLOv5, we only get one tensor and then need to process that. This comment, FOR YOLOv8,
        # might be relevant, https://github.com/ultralytics/ultralytics/issues/2950 BUT YOLOv5 and
        # is different than YOLOv8.
        #
        # In this case YOLOv5 returns [1, 6300, 85]. THE OUTPUT CAN BE DIFFERENT BETWEEN DIFFERENT
        # YOLOv5 USES BASED ON HOW THE ORIGINAL MODEL WAS CONVERTED.
        #
        # POTENTIALLY THIS IS:
        #
        # Batch Size (1): The first dimension represents the batch size, which is 1 in this case.
        # This means the model processes one image at a time.
        #
        # Number of Predictions (6300): The second dimension (6300) represents the total number of
        # bounding box predictions made by the model. This number is derived from the model's architecture
        # and input size.
        #
        # Prediction Information (85): The third dimension (85) contains the information for each predicted
        # bounding box. It's MAY BE broken down as follows:
        #    - 4 values for bounding box coordinates (x, y, width, height)
        #    - 1 value for objectness score. From a Perplexity search: "The detection scores, also known
        #      as confidence scores, indicate how confident the model is about the presence of an object
        #      in a particular bounding box"
        #    - 80 values for class probabilities (assuming the model was trained on COCO dataset with
        #      80 classes).  From a Perplexity search: "YOLOv5s is typically trained on the COCO dataset,
        #      which has 80 object classes. For each detected object, the model outputs class
        #      probabilities for all 80 classes. The class with the highest probability is assigned to
        #      that detection."
        #
        # This Perplexity search seemed to turn up useful information:
        # https://www.perplexity.ai/search/for-yolov5s-explain-the-output-ZY9NjbmpRWyxcoXko_X.3g
        #
        #
        # Interesting quote "It appears that the issue you're encountering with the output order after
        # applying NMS is related to a known behavior in TensorFlow Lite." From
        # https://github.com/ultralytics/yolov5/issues/1981#issuecomment-1893230350. And see this
        # https://github.com/tensorflow/tensorflow/issues/33303#issuecomment-1068201819
        #
        # The num_detections you're seeing is related to the total number of predictions made by the
        # model before any filtering. To control the number of detections, you'll need to apply
        # Non-Maximum Suppression (NMS) and a confidence threshold to filter out less likely
        # predictions and reduce the number of final detections.
        #
        # Here's a brief outline of what you need to do:
        #
        # - Apply a Confidence Threshold: Filter out detections that have a confidence score lower
        # than a certain threshold. This will remove a lot of low-confidence predictions.
        #
        # - Apply NMS: Use Non-Maximum Suppression to eliminate overlapping boxes. This will keep
        # only the best bounding box when multiple boxes are detected for the same object.
        #
        # - Limit the Number of Detections: After applying the confidence threshold and NMS, you
        # can sort the remaining detections by confidence and keep only the top N detections, where
        # N is the number you want (e.g., 10).
        #
        # This post-processing needs to be done after you run inference with the TFLite model and before
        # you interpret the results. If you want to integrate this directly into the TFLite model, it
        # would require custom operations, which is quite complex.
        #
        # From https://github.com/ultralytics/yolov5/issues/1981#issuecomment-1886680620

        output_data = interpreter.get_tensor(output_details[0]["index"])

        # print()
        # print("Output data")
        # print("Shape of output data:", output_data.shape)
        # pprint(output_data)
        # print()

        boxes = output_data[..., :4]  # Results in an array of (1, 6300, 4)
        scores = output_data[..., 4]  # Results in an arragy of (1, 6300)
        class_ids = output_data[..., 5:]  # results in an array of (1, 6300, 80)

        # print("First three boxes:")
        # box_shape = boxes.shape[1]
        # for i in range(min(3, box_shape)):
        #     print(f"Box {i}: {boxes[0, i]} score {scores[0, i]}")

        # Apply non-max suppression
        max_output_size = 100
        iou_threshold = 0.5
        score_threshold = 0.5

        selected_indices = tf.image.non_max_suppression(
            boxes[0],
            scores[0],
            max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        ) 
        # In my test case, returns an tensor of shape (62,). That is indices of the "best"
        # bounding boxes that meet the criteria. So there should be 62 boxes for this one image.
        # It does seem to be sorted in decreasing order on scores. This sorting doesn't take into
        # account classes.
        #
        # See https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression

        detection_boxes = tf.gather(boxes[0], selected_indices)
        detection_scores = tf.gather(scores[0], selected_indices)
        detection_classes = tf.gather(class_ids[0], selected_indices)
        # In my test case, returns:
        # detection_boxes is a TENSOR of shape (62, 4)
        # detection_scores is a TENSOR of shape (62)
        # detection_classes is a TENSOR of shape (62, 80)
        #
        # See https://www.tensorflow.org/api_docs/python/tf/gather

        # print (f"Detection boxes shape {detection_boxes.shape}")
        # print("First n detection boxes:")
        # for i in range(min(3, detection_boxes.shape[0])):
        #     print(f"Box {i} - score: {detection_scores[i]} classes: {detection_classes[i].numpy()}")
        #     # print(f"Box {i}: {detection_boxes[i].numpy()} score: {detection_scores[i]}")


        num_boxes = int(detection_boxes.shape[0])

        # print(f"Detection boxes shape: {detection_boxes.shape}")
        # print(f"Detection classes shape: {detection_classes.shape}")
        # print(f"Detection scores shape: {detection_scores.shape}")

        # Further filter results.
        #
        # detect_boxes is OK, but probably needs to be further filtered.  Ideally, Only the objects with
        # a detection_score * max(detection_class) > some threshold should be included. But that's
        # for the classes we're trained on. So maybe not? In my test case, we don't want to return
        # 63 boxes!
        #
        # detection_scores[i] probably needs to be detection_score * max(detection_class).
        #
        # detection_classes[i] probably needs to be the index of the maximum value.

        # A basic loop through everything:

        final_boxes = np.empty((0, 4))
        final_scores = []
        final_classes = []

        for i in range(num_boxes):

            if detection_scores[i] == 1.0:  # Arbitrary value for now
                # Find the max class
                max_class_score = 0.0
                selected_index = 0
                for j in range(detection_classes.shape[1]):
                    # Should loop through 80 items
                    if detection_classes[i][j] > max_class_score:
                        max_class_score = detection_classes[i][j].numpy()
                        selected_index = j

            if (detection_scores[i] * max_class_score > 0.5):
                box = tuple(detection_boxes[i].numpy())
                # I believe that YOLOv5 returns center and height/width as the rectnagle
                x, y, w, h = box[0], box[1], box[2], box[3]  # xywh
                xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy
                final_boxes = np.vstack((final_boxes,  xyxy))
                final_scores.append(detection_scores[i] * max_class_score)
                final_classes.append(selected_index)
    
        final_num_boxes = len(final_scores)

        print(f"Final boxes shape: {np.array(final_boxes).shape}")
        print("First n detection boxes:")
        for i in range(min(100, len(final_scores))):
            print(f"{i}: detection_score: {detection_scores[i]} final_score: {final_scores[i]} class: {final_classes[i]}")
            # print(f"Box {i}: {final_boxes[i]} score: {final_scores[i]}")


        # TODO: Look at the YOLODetect in the non_working_tflite_yolov5.py.  Does that do what I'm trying to do? Does it give a clue?

        return final_boxes, final_scores, final_classes, final_num_boxes
