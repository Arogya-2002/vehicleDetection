import os
import sys
import pandas as pd
from paddleocr import PaddleOCR
from datetime import datetime
from src.exceptions.exceptions import CustomException
from src.logs.logger import logging


class BatchOcrProcessor:
    def __init__(self, artifacts_root_dir: str, output_csv_path: str):
        self.artifacts_root_dir = artifacts_root_dir
        self.output_csv_path = output_csv_path
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logging.info("PaddleOCR model initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def safe_ocr(self, image_path):
        """
        Safely run OCR on given image path.
        Returns list of detected boxes or empty list if failed or no detection.
        """
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if result and len(result) > 0 and result[0]:
                logging.debug(f"OCR detected {len(result[0])} boxes in {image_path}")
                return result[0]
            else:
                logging.warning(f"No OCR result for image: {image_path}")
                return []
        except Exception as e:
            logging.error(f"OCR failed for image {image_path}: {e}")
            return []

    def process_all(self):
        try:
            if not os.path.exists(self.artifacts_root_dir):
                raise FileNotFoundError(f"Artifacts directory not found: {self.artifacts_root_dir}")

            records = []
            logging.info(f"Starting OCR batch processing in directory: {self.artifacts_root_dir}")

            for root, dirs, files in os.walk(self.artifacts_root_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(root, file)
                        run_folder = os.path.relpath(root, self.artifacts_root_dir)

                        logging.info(f"Processing image: {image_path}")
                        ocr_results = self.safe_ocr(image_path)

                        for line in ocr_results:
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
                logging.warning("No text detected in any images. No CSV will be generated.")
                return

            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
            pd.DataFrame(records).to_csv(self.output_csv_path, index=False)

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
