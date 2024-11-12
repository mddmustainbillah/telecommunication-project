from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import mlflow
from mlflow.tracking import MlflowClient
from telecommunication.config import logger, PROJ_ROOT
from telecommunication.modeling.predict import predict
from typing import Optional  # Added import for Optional

# Initialize FastAPI app
app = FastAPI(title="Telecommunication Commission Prediction API")

# Initialize MLflow client
MLRUNS_DIR = PROJ_ROOT / "mlruns"
os.environ['MLFLOW_TRACKING_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
os.environ['MLFLOW_REGISTRY_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

client = MlflowClient()

class PredictionInput(BaseModel):
    bundle_type: str       # Changed to str for mapping
    operator: str          # Changed to str for mapping
    validity: float        # Already float
    regular_price: float   # Already float
    selling_price: float   # Already float
    internet: float        # Already float
    minutes: float         # Already float

class PredictionOutput(BaseModel):
    predicted_commission: float
    status: str
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Telecommunication Commission Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    """Make a single prediction"""
    try:
        # Convert input data to dictionary
        input_data_dict = input_data.dict()
        
        # Call the predict function from predict.py
        result = predict(input_data_dict)

        if result["status"] == "success":
            return PredictionOutput(
                predicted_commission=result["predictions"],
                status="success"
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return PredictionOutput(
            predicted_commission=0.0,
            status="failed",
            error=str(e)
        )

# Example usage with Python:
"""
import requests

url = "http://localhost:8000/predict"
data = {
    "bundle_type": "Internet & Minute",  # str
    "operator": "gp",                     # str
    "validity": 30.0,                     # float
    "regular_price": 100.0,               # float
    "selling_price": 90.0,                # float
    "internet": 2.0,                      # float
    "minutes": 100.0                      # float
}

response = requests.post(url, json=data)
print(response.json())
"""