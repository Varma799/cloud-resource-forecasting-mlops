import os
import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, load_object
from src.components.scaling_recommender import ScalingRecommender


class PredictionPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = read_yaml(config_path)
        self.model_path = os.path.join(self.config["paths"]["model_dir"], "best_model.pkl")
        self.model = load_object(self.model_path)
        self.recommender = ScalingRecommender()

    def predict(self, input_data: dict) -> dict:
        try:
            logger.info("Starting prediction pipeline")

            df = pd.DataFrame([input_data])

            prediction = float(self.model.predict(df)[0])

            predicted_memory = float(input_data.get("memory_usage", 0))
            scaling_action = self.recommender.recommend(prediction, predicted_memory)

            result = {
                "predicted_future_cpu_usage": round(prediction, 2),
                "predicted_future_memory_usage": round(predicted_memory, 2),
                "scaling_action": scaling_action
            }

            logger.info(f"Prediction result: {result}")
            return result

        except Exception as e:
            logger.error("Error occurred during prediction")
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_input = {
        "cpu_usage": 52.0,
        "memory_usage": 66.0,
        "disk_io": 140.0,
        "network_in": 310.0,
        "network_out": 290.0,
        "request_count": 1650,
        "hour": 10,
        "day_of_week": 3,
        "is_weekend": 0,
        "cpu_lag_1": 50.0,
        "memory_lag_1": 64.0,
        "request_count_lag_1": 1600,
        "cpu_lag_2": 48.0,
        "memory_lag_2": 63.0,
        "request_count_lag_2": 1550,
        "cpu_rolling_mean_2": 51.0,
        "memory_rolling_mean_2": 65.0,
        "cpu_rolling_mean_3": 49.5,
        "memory_rolling_mean_3": 64.3
    }

    pipeline = PredictionPipeline()
    print(pipeline.predict(sample_input))