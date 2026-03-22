from fastapi import FastAPI
from app.schemas import PredictionRequest
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Cloud Resource Forecasting API")

pipeline = PredictionPipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    result = pipeline.predict(request.model_dump())
    return result