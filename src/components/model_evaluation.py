import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml, ensure_directory, load_object


class ModelEvaluation:
    def __init__(self, config_path: str = "config/config.yaml", params_path: str = "config/params.yaml"):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

    def initiate_model_evaluation(self) -> str:
        try:
            transformed_data_path = self.config["paths"]["transformed_data"]
            model_path = os.path.join(self.config["paths"]["model_dir"], "best_model.pkl")
            reports_dir = self.config["paths"]["reports_dir"]
            plots_dir = self.config["paths"]["plots_dir"]

            target_column = self.params["data"]["target_cpu"]
            test_size = self.params["data"]["test_size"]
            random_seed = self.config["project"]["random_seed"]

            logger.info("Starting model evaluation")

            df = pd.read_csv(transformed_data_path)

            columns_to_drop = [
                "timestamp",
                "server_id",
                "future_cpu_usage",
                "future_memory_usage",
                "scaling_action"
            ]

            X = df.drop(columns=columns_to_drop, errors="ignore")
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )

            model = load_object(model_path)
            predictions = model.predict(X_test)

            metrics = {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
                "r2": float(r2_score(y_test, predictions))
            }

            ensure_directory(reports_dir)
            ensure_directory(plots_dir)

            evaluation_report_path = os.path.join(reports_dir, "evaluation_report.json")
            with open(evaluation_report_path, "w") as report_file:
                json.dump(metrics, report_file, indent=4)

            actual_vs_predicted_path = os.path.join(plots_dir, "actual_vs_predicted.png")
            plt.figure(figsize=(8, 5))
            plt.scatter(y_test, predictions)
            plt.xlabel("Actual Future CPU Usage")
            plt.ylabel("Predicted Future CPU Usage")
            plt.title("Actual vs Predicted CPU Usage")
            plt.tight_layout()
            plt.savefig(actual_vs_predicted_path)
            plt.close()

            residuals_path = os.path.join(plots_dir, "residual_plot.png")
            residuals = y_test - predictions
            plt.figure(figsize=(8, 5))
            plt.scatter(predictions, residuals)
            plt.axhline(y=0)
            plt.xlabel("Predicted Future CPU Usage")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")
            plt.tight_layout()
            plt.savefig(residuals_path)
            plt.close()

            logger.info(f"Evaluation report saved to: {evaluation_report_path}")
            logger.info(f"Plots saved to: {plots_dir}")

            return evaluation_report_path

        except Exception as e:
            logger.error("Error occurred during model evaluation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    print(evaluator.initiate_model_evaluation())