import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml


class ScalingRecommender:
    def __init__(self, params_path: str = "config/params.yaml"):
        self.params = read_yaml(params_path)
        self.scale_up_cpu = self.params["scaling_thresholds"]["scale_up_cpu"]
        self.scale_up_memory = self.params["scaling_thresholds"]["scale_up_memory"]
        self.scale_down_cpu = self.params["scaling_thresholds"]["scale_down_cpu"]
        self.scale_down_memory = self.params["scaling_thresholds"]["scale_down_memory"]

    def recommend(self, predicted_cpu: float, predicted_memory: float) -> str:
        try:
            if predicted_cpu > self.scale_up_cpu or predicted_memory > self.scale_up_memory:
                return "scale_up"
            if predicted_cpu < self.scale_down_cpu and predicted_memory < self.scale_down_memory:
                return "scale_down"
            return "stable"
        except Exception as e:
            logger.error("Error occurred during scaling recommendation")
            raise CustomException(e, sys)

    def add_recommendation_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Adding scaling recommendation column")
            df["scaling_action"] = df.apply(
                lambda row: self.recommend(
                    row["future_cpu_usage"],
                    row["future_memory_usage"]
                ),
                axis=1
            )
            return df
        except Exception as e:
            logger.error("Error occurred while adding recommendation column")
            raise CustomException(e, sys)