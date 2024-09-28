import torch
from PIL import Image
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

IMG_DIR = "/Users/todd/Documents/dev/images/samples/2"


class RunnerModelTH:
 
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def process_image(self, image_path):
        img = Image.open(image_path)

        # Perform inference
        results = self.model(img)

        # Print results
        # results.print()

        # Show results
        # results.show()

        # Save results
        # results.save()

        # Optionally, you can also get the results as a pandas DataFrame
        df = results.pandas().xyxy[0]

        detections = []
        for _, row in df.iterrows():
            detection = {
            'name': row['name'],
            'confidence': row['confidence'],
            'rectangle': {
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax']
            }
            }
            detections.append(detection)

        return detections

    def process_directory(self):
        stats = []

        for filename in os.listdir(IMG_DIR):
            img_path = os.path.join(IMG_DIR, filename)
            datum = self.process_image(img_path)

            stats.append(datum)

        cnt = 0
        for detections in stats:
            if len(detections) > 0:
                cnt += 1

        # for detections in stats:
        #     print(str(detections))

        print(f"num_files {len(stats)} with_objects {cnt}")


if __name__ == "__main__":
    runner = RunnerModelTH().process_directory()
