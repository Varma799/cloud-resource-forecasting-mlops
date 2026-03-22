from pydantic import BaseModel


class PredictionRequest(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_in: float
    network_out: float
    request_count: int
    hour: int
    day_of_week: int
    is_weekend: int
    cpu_lag_1: float
    memory_lag_1: float
    request_count_lag_1: int
    cpu_lag_2: float
    memory_lag_2: float
    request_count_lag_2: int
    cpu_rolling_mean_2: float
    memory_rolling_mean_2: float
    cpu_rolling_mean_3: float
    memory_rolling_mean_3: float