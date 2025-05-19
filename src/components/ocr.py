import os
import json
import sys
import pandas as pd
from datetime import datetime
from paddleocr import PaddleOCR
from src.exceptions.exceptions import CustomException
from src.loggs.logger import logging


class BatchOcrProcessor:
    def __init__(self, artifacts_root_dir: str, output_csv_path: str, state_file: str = ".ocr_state.json"):
        self.artifacts_root_dir = artifacts_root_dir
        self.output_csv_path = output_csv_path
        self.state_file = state_file
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.last_processed_folder = self._load_state()

    def _load_state(self) -> str:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f).get("last_folder", "")
            except Exception as e:
                logging.warning(f"Failed to load OCR state file: {e}")
        return ""

    def _save_state(self, folder_name: str):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"last_folder": folder_name}, f)
                logging.info(f"OCR state saved: last_folder={folder_name}")
        except Exception as e:
            logging.warning(f"Failed to save OCR state: {e}")

    def _safe_ocr(self, image_path):
        try:
            result = self.ocr.ocr(image_path, cls=True)
            return result[0] if result and isinstance(result, list) and result[0] else []
        except Exception as e:
            logging.error(f"OCR failed for {image_path}: {e}")
            return []

    def process_all(self):
        try:
            records = []
            logging.info(f"Starting OCR processing from: {self.artifacts_root_dir}")

            all_dirs = sorted([
                os.path.join(self.artifacts_root_dir, d)
                for d in os.listdir(self.artifacts_root_dir)
                if os.path.isdir(os.path.join(self.artifacts_root_dir, d))
            ])

            for dir_path in all_dirs:
                run_folder = os.path.basename(dir_path)

                if self.last_processed_folder and run_folder <= self.last_processed_folder:
                    continue

                logging.info(f"Processing folder: {run_folder}")

                for file in os.listdir(dir_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(dir_path, file)
                        ocr_lines = self._safe_ocr(image_path)

                        for line in ocr_lines:
                            text, conf = line[1]
                            records.append({
                                "run_folder": run_folder,
                                "image_file": file,
                                "extracted_text": text,
                                "confidence": conf,
                                "processed_at": datetime.now().isoformat()
                            })

                self._save_state(run_folder)

            if not records:
                logging.info("OCR processing complete. No new records to append.")
                return

            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
            df_new = pd.DataFrame(records)

            if os.path.exists(self.output_csv_path):
                df_new.to_csv(self.output_csv_path, mode='a', index=False, header=False)
            else:
                df_new.to_csv(self.output_csv_path, index=False)

            logging.info(f"OCR processing complete. Results appended to: {self.output_csv_path}")

        except Exception as e:
            logging.error(f"Error during OCR processing: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.constants import ARTIFACTS_DIR, OCR_OUTPUT_CSV

    processor = BatchOcrProcessor(
        artifacts_root_dir=ARTIFACTS_DIR,
        output_csv_path=OCR_OUTPUT_CSV
    )
    processor.process_all()
