import os
import sys
import json
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, ensure_directory


class MonitoringPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = read_yaml(config_path)

    def run_monitoring_check(self) -> str:
        try:
            transformed_data_path = self.config["paths"]["transformed_data"]
            reports_dir = self.config["paths"]["reports_dir"]

            logger.info("Starting monitoring pipeline")

            df = pd.read_csv(transformed_data_path)

            monitoring_summary = {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "cpu_usage_mean": float(df["cpu_usage"].mean()),
                "memory_usage_mean": float(df["memory_usage"].mean()),
                "future_cpu_usage_mean": float(df["future_cpu_usage"].mean()),
                "future_memory_usage_mean": float(df["future_memory_usage"].mean())
            }

            ensure_directory(reports_dir)
            monitoring_report_path = os.path.join(reports_dir, "monitoring_report.json")

            with open(monitoring_report_path, "w") as report_file:
                json.dump(monitoring_summary, report_file, indent=4)

            logger.info(f"Monitoring report saved to: {monitoring_report_path}")
            return monitoring_report_path

        except Exception as e:
            logger.error("Error occurred during monitoring pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    print(pipeline.run_monitoring_check())