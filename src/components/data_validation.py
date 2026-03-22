import os
import sys
import json
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, ensure_directory


class DataValidation:
    def __init__(self, config_path: str = "config/config.yaml", schema_path: str = "config/schema.yaml"):
        self.config = read_yaml(config_path)
        self.schema = read_yaml(schema_path)

    def validate_data(self) -> bool:
        try:
            ingested_data_path = self.config["paths"]["ingested_data"]
            reports_dir = self.config["paths"]["reports_dir"]
            validation_report_path = os.path.join(reports_dir, "validation_report.json")

            logger.info("Starting data validation")
            logger.info(f"Reading ingested data from: {ingested_data_path}")

            if not os.path.exists(ingested_data_path):
                raise FileNotFoundError(f"Ingested data file not found at path: {ingested_data_path}")

            df = pd.read_csv(ingested_data_path)

            required_columns = self.schema["required_columns"]

            validation_report = {
                "missing_columns": [],
                "duplicate_rows": int(df.duplicated().sum()),
                "null_counts": df.isnull().sum().to_dict(),
                "invalid_timestamps": 0,
                "negative_values": {},
                "status": "success"
            }

            missing_columns = [col for col in required_columns if col not in df.columns]
            validation_report["missing_columns"] = missing_columns

            parsed_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            validation_report["invalid_timestamps"] = int(parsed_timestamps.isnull().sum())

            numeric_columns = ["cpu_usage", "memory_usage", "disk_io", "network_in", "network_out", "request_count"]
            for col in numeric_columns:
                validation_report["negative_values"][col] = int((df[col] < 0).sum())

            if (
                missing_columns
                or validation_report["duplicate_rows"] > 0
                or validation_report["invalid_timestamps"] > 0
                or any(count > 0 for count in validation_report["null_counts"].values())
                or any(count > 0 for count in validation_report["negative_values"].values())
            ):
                validation_report["status"] = "failed"

            ensure_directory(reports_dir)
            with open(validation_report_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logger.info(f"Validation report saved to: {validation_report_path}")
            logger.info(f"Validation status: {validation_report['status']}")

            return validation_report["status"] == "success"

        except Exception as e:
            logger.error("Error occurred during data validation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    validator = DataValidation()
    print(validator.validate_data())