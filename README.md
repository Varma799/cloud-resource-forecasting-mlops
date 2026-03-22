# Cloud Resource Demand Forecasting and Auto Scaling Recommendation System

## Project Overview

This project forecasts short-term cloud resource demand using historical infrastructure metrics and recommends scaling actions for operations teams. I built it as an end-to-end machine learning workflow that covers data ingestion, validation, feature engineering, model training, evaluation, API serving, and local testing.

## Why I Built This

Cloud systems rarely fail in neat, predictable ways. CPU usage spikes, traffic bursts, and memory pressure can build quickly, especially when workloads change across services. I wanted to build a project that feels closer to a real operational problem than a generic notebook exercise.

Instead of only predicting a number, this system also translates model output into an action:
scale_up
stable
scale_down

That makes the project more useful for infrastructure decision support and closer to how engineering teams actually think.

## Business Problem

Operations teams need a way to anticipate resource pressure before it impacts performance or cost. Static threshold alerts are reactive. This project uses historical cloud usage patterns to forecast future CPU demand and recommend scaling actions in advance.

## Project Goals

Build a reproducible ML pipeline for cloud resource forecasting
Validate and transform infrastructure usage data
Train and compare multiple regression models
Generate scaling recommendations from prediction outputs
Expose predictions through a FastAPI service
Add automated tests for API and processed data quality
Create evaluation artifacts that can be reviewed visually

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Pytest
- Matplotlib
- YAML
- Pickle

## Project Structure

```text
cloud-resource-forecasting-mlops/
├── app/
├── config/
├── data/
├── artifacts/
├── logs/
├── notebooks/
├── src/
├── tests/
├── generate_sample_data.py
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── dvc.yaml

## Pipeline Flow 

Generate or load raw cloud resource usage data
Ingest and store processed raw input
Validate schema, missing values, duplicates, and invalid timestamps
Create time-based, lag-based, and rolling features
Train and compare multiple forecasting models
Select and save the best model
Evaluate predictions and generate plots
Serve predictions through FastAPI
Test endpoints and processed data with pytest

## Features Engineered 

- hour
- day_of_week
- is_weekend
- cpu_lag_1
- cpu_lag_2
- memory_lag_1
- memory_lag_2
- request_count_lag_1
- request_count_lag_2
- cpu_rolling_mean_2
- cpu_rolling_mean_3
- memory_rolling_mean_2
- memory_rolling_mean_3

##  Modeling Approach

The project predicts future CPU usage using regression models trained on transformed infrastructure metrics. I compared:
Linear Regression
Random Forest Regressor
XGBoost Regressor

The best model is selected automatically and saved for downstream inference.

## Scaling Logic

The forecasting output is converted into an operational recommendation using rule-based thresholds.

scale_up when predicted usage crosses upper CPU or memory thresholds
scale_down when both CPU and memory are below lower thresholds
stable otherwise

## API Endpoints

GET /health
Returns service health status

POST /predict
Accepts cloud resource metrics and engineered features, then returns:
predicted_future_cpu_usage
predicted_future_memory_usage
scaling_action

## Example Result

```json
{
  "predicted_future_cpu_usage": 53.29,
  "predicted_future_memory_usage": 66.0,
  "scaling_action": "stable"
}
```

## Evaluation Artifacts

The project generates:
model_report.json
evaluation_report.json
actual_vs_predicted.png
residual_plot.png
validation_report.json

## Testing

This project includes automated tests for:
health endpoint
prediction endpoint
processed data file existence
required transformed columns

## Current Status

Completed local end-to-end workflow:
data generation
ingestion
validation
transformation
training
evaluation
prediction pipeline
FastAPI app
Swagger testing
pytest validation

## Future Improvements

- Add MLflow experiment tracking
- Add Docker-based deployment
- Add DVC for dataset and model versioning
- Create CI workflow with GitHub Actions
- Expand prediction support for memory forecasting as a second trained target
- Deploy the API to a cloud service
- Extend the current local monitoring summary into feature drift and prediction drift tracking

## How to Run Locally

- Activate the virtual environment
- Generate sample data
- Run the training pipeline
- Start the API server
- Open Swagger docs and test the endpoints

## Commands used:

```bash
python generate_sample_data.py
python -m src.pipeline.training_pipeline
python -m uvicorn app.main:app --reload
python -m pytest
```
## Why This Project Adds Value to My Portfolio

This project demonstrates more than model training. It shows that I can structure a machine learning system end to end, validate data, engineer features, compare models, expose predictions through an API, and test the workflow locally. It also reflects the type of cloud and operational use cases that align with my broader engineering profile.