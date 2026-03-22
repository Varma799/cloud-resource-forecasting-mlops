import os
import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, ensure_directory


class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = read_yaml(config_path)

    def initiate_data_ingestion(self) -> str:
        try:
            raw_data_path = self.config["paths"]["raw_data"]
            ingested_data_path = self.config["paths"]["ingested_data"]

            logger.info("Starting data ingestion")
            logger.info(f"Reading raw data from: {raw_data_path}")

            if not os.path.exists(raw_data_path):
                raise FileNotFoundError(f"Raw data file not found at path: {raw_data_path}")

            df = pd.read_csv(raw_data_path)

            ensure_directory(os.path.dirname(ingested_data_path))
            df.to_csv(ingested_data_path, index=False)

            logger.info(f"Raw data loaded successfully with shape: {df.shape}")
            logger.info(f"Ingested data saved to: {ingested_data_path}")

            return ingested_data_path

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    print(ingestion.initiate_data_ingestion())