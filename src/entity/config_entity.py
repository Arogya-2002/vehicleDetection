from src.exceptions.exceptions import CustomException
from src.loggs.logger import logging
from src.constants import *
import os, sys
from datetime import datetime

class TrainerConfig:

    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_path = MODEL_PATH
        self.cropped_image_dir = CROPPED_IMAGE_DIR   # <-- Only dir path like 'Artifacts/'
        self.data = DATA
        self.cropped_image_name = CROPPED_IMAGE_NAME # <-- Only file name like 'cropped_licence_plate.jpg'


class PlateDetectionConfig:
    def __init__(self, trainer_config: TrainerConfig):
        self.image_path = os.path.join(trainer_config.data, 'numberplate1.webp')
        self.model_file_path = os.path.join(trainer_config.model_path, trainer_config.model_name)

        # Generate unique run folder with date-time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_dir = os.path.join(trainer_config.cropped_image_dir, f"run_{timestamp}")

        # Ensure the directory exists
        os.makedirs(self.run_dir, exist_ok=True)

        self.cropped_image_dir = self.run_dir  # Redirect all crops to this folder


class OcrConfig:
    def __init__(self, trainer_config: TrainerConfig):
        self.cropped_image_dir = trainer_config.cropped_image_dir



