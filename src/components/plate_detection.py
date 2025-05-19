from src.exceptions.exceptions import CustomException
from src.entity.config_entity import PlateDetectionConfig, TrainerConfig
from src.entity.artifact_entity import PlateDetectionArtifact
from src.loggs.logger import logging
import os, sys
import cv2
from ultralytics import YOLO

class PlateDetection:
    def __init__(self):
        self.plate_detection_config = PlateDetectionConfig(trainer_config=TrainerConfig())

    def convertBGRtoRGB(self):
        try:
            img = cv2.imread(self.plate_detection_config.image_path)
            if img is None:
                raise ValueError(f"Failed to read image at {self.plate_detection_config.image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            raise CustomException(e, sys)

    def detect_plate(self, img_rgb):
        try:
            model = YOLO(self.plate_detection_config.model_file_path, task="detect")
            results = model(img_rgb)
            logging.info(f"Model prediction on the image: {results}")

            num_boxes = len(results[0].boxes)
            logging.info(f"Number of detected license plates: {num_boxes}")

            if num_boxes == 0:
                logging.warning("No license plates detected.")

            return results
        except Exception as e:
            raise CustomException(e, sys)

    def extract_box_coord(self, results, img_rgb):
        try:
            xywh = results[0].boxes.xywh
            conf = results[0].boxes.conf
            cls = results[0].boxes.cls

            num_boxes = len(xywh)
            if num_boxes == 0:
                raise ValueError("No bounding boxes found in detection results.")

            logging.info(f"Number of license plates detected: {num_boxes}")

            os.makedirs(self.plate_detection_config.cropped_image_dir, exist_ok=True)
            cropped_image_paths = []

            for idx, box in enumerate(xywh):
                x_center, y_center, width, height = box
                logging.info(f"Plate {idx+1} - Bounding box: {x_center}, {y_center}, {width}, {height}")

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                img_height, img_width, _ = img_rgb.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                cropped_plate = img_rgb[y1:y2, x1:x2]

                cropped_image_file_path = os.path.join(self.plate_detection_config.cropped_image_dir, f"cropped_plate_{idx+1}.jpg")

                cv2.imwrite(cropped_image_file_path, cropped_plate)
                logging.info(f"Cropped plate {idx+1} saved at: {cropped_image_file_path}")

                cropped_image_paths.append(cropped_image_file_path)

            return PlateDetectionArtifact(cropped_image_path=cropped_image_paths)
        except Exception as e:
            raise CustomException(f"Error in extract_box_coord: {e}", sys)

if __name__ == "__main__":
    plate_detection = PlateDetection()
    logging.info("Starting Plate Detection Pipeline")

    img_rgb = plate_detection.convertBGRtoRGB()
    results = plate_detection.detect_plate(img_rgb)
    artifact = plate_detection.extract_box_coord(results, img_rgb)

    logging.info(f"Final cropped images saved: {artifact.cropped_image_path}")
