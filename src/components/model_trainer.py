import os
import sys
import json
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, ensure_directory, save_object


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml", params_path: str = "config/params.yaml"):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

    def evaluate_model(self, y_true, y_pred) -> dict:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)

        try:
            r2 = float(r2_score(y_true, y_pred))
            if math.isnan(r2):
                r2 = None
        except Exception:
            r2 = None

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

    def initiate_model_training(self) -> str:
        try:
            transformed_data_path = self.config["paths"]["transformed_data"]
            model_dir = self.config["paths"]["model_dir"]
            reports_dir = self.config["paths"]["reports_dir"]

            test_size = self.params["data"]["test_size"]
            target_column = self.params["data"]["target_cpu"]

            logger.info("Starting model training")
            logger.info(f"Reading transformed data from: {transformed_data_path}")

            df = pd.read_csv(transformed_data_path)

            if df.empty:
                raise ValueError("Transformed dataset is empty. Cannot train model.")

            columns_to_drop = [
                "timestamp",
                "server_id",
                "future_cpu_usage",
                "future_memory_usage",
                "scaling_action"
            ]

            X = df.drop(columns=columns_to_drop, errors="ignore")
            y = df[target_column]

            if len(df) < 3:
                raise ValueError("Not enough rows in transformed dataset to train and evaluate models.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.config["project"]["random_seed"]
            )

            models = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config["project"]["random_seed"]
                ),
                "xgboost": XGBRegressor(
                    n_estimators=self.params["model"]["n_estimators"],
                    learning_rate=self.params["model"]["learning_rate"],
                    max_depth=self.params["model"]["max_depth"],
                    subsample=self.params["model"]["subsample"],
                    colsample_bytree=self.params["model"]["colsample_bytree"],
                    random_state=self.config["project"]["random_seed"]
                )
            }

            model_report = {}
            best_model_name = None
            best_model = None
            best_mae = float("inf")

            for model_name, model in models.items():
                logger.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                metrics = self.evaluate_model(y_test, predictions)

                model_report[model_name] = metrics

                if metrics["mae"] < best_mae:
                    best_mae = metrics["mae"]
                    best_model_name = model_name
                    best_model = model

            ensure_directory(model_dir)
            ensure_directory(reports_dir)

            best_model_path = os.path.join(model_dir, "best_model.pkl")
            report_path = os.path.join(reports_dir, "model_report.json")

            save_object(best_model_path, best_model)

            final_report = {
                "best_model": best_model_name,
                "metrics": model_report
            }

            with open(report_path, "w") as report_file:
                json.dump(final_report, report_file, indent=4)

            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best model saved to: {best_model_path}")
            logger.info(f"Model report saved to: {report_path}")

            return best_model_path

        except Exception as e:
            logger.error("Error occurred during model training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    print(trainer.initiate_model_training())