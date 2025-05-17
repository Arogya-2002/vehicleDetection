import os
import pandas as pd
from paddleocr import PaddleOCR
from datetime import datetime
from src.exceptions.exceptions import CustomException
from src.logs.logger import logging
import sys

class BatchOcrProcessor:
    def __init__(self, artifacts_root_dir: str, output_csv_path: str):
        self.artifacts_root_dir = artifacts_root_dir
        self.output_csv_path = output_csv_path
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')

    def process_all(self):
        try:
            records = []

            logging.info(f"Starting OCR batch processing in directory: {self.artifacts_root_dir}")

            for root, dirs, files in os.walk(self.artifacts_root_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(root, file)
                        run_folder = os.path.relpath(root, self.artifacts_root_dir)

                        logging.info(f"Processing image: {image_path}")
                        result = self.ocr.ocr(image_path, cls=True)

                        for line in result[0]:
                            text = line[1][0]
                            conf = line[1][1]
                            records.append({
                                "run_folder": run_folder,
                                "image_file": file,
                                "extracted_text": text,
                                "confidence": conf,
                                "processed_at": datetime.now().isoformat()
                            })

            if not records:
                logging.warning("No text detected in any images.")
                return

            df = pd.DataFrame(records)
            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
            df.to_csv(self.output_csv_path, index=False)

            logging.info(f"OCR batch processing completed. Results saved to: {self.output_csv_path}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.constants import ARTIFACTS_DIR, OCR_OUTPUT_CSV

    processor = BatchOcrProcessor(
        artifacts_root_dir=ARTIFACTS_DIR,
        output_csv_path=OCR_OUTPUT_CSV
    )
    processor.process_all()