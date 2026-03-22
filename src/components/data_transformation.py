import os
import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, ensure_directory
from src.components.scaling_recommender import ScalingRecommender


class DataTransformation:
    def __init__(self, config_path: str = "config/config.yaml", params_path: str = "config/params.yaml"):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.recommender = ScalingRecommender(params_path=params_path)

    def initiate_data_transformation(self) -> str:
        try:
            ingested_data_path = self.config["paths"]["ingested_data"]
            transformed_data_path = self.config["paths"]["transformed_data"]

            lag_steps = self.params["features"]["lag_steps"]
            rolling_windows = self.params["features"]["rolling_windows"]

            logger.info("Starting data transformation")
            logger.info(f"Reading ingested data from: {ingested_data_path}")

            df = pd.read_csv(ingested_data_path)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by=["server_id", "timestamp"]).reset_index(drop=True)

            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            for lag in lag_steps:
                df[f"cpu_lag_{lag}"] = df.groupby("server_id")["cpu_usage"].shift(lag)
                df[f"memory_lag_{lag}"] = df.groupby("server_id")["memory_usage"].shift(lag)
                df[f"request_count_lag_{lag}"] = df.groupby("server_id")["request_count"].shift(lag)

            for window in rolling_windows:
                df[f"cpu_rolling_mean_{window}"] = (
                    df.groupby("server_id")["cpu_usage"]
                    .transform(lambda x: x.rolling(window=window).mean())
                )
                df[f"memory_rolling_mean_{window}"] = (
                    df.groupby("server_id")["memory_usage"]
                    .transform(lambda x: x.rolling(window=window).mean())
                )

            df["future_cpu_usage"] = df.groupby("server_id")["cpu_usage"].shift(-1)
            df["future_memory_usage"] = df.groupby("server_id")["memory_usage"].shift(-1)

            df = df.dropna().reset_index(drop=True)

            df = self.recommender.add_recommendation_column(df)

            ensure_directory(os.path.dirname(transformed_data_path))
            df.to_csv(transformed_data_path, index=False)

            logger.info(f"Transformed data saved to: {transformed_data_path}")
            logger.info(f"Transformed dataframe shape: {df.shape}")

            return transformed_data_path

        except Exception as e:
            logger.error("Error occurred during data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    print(transformer.initiate_data_transformation())
