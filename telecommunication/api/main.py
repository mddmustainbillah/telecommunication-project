from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, List, Optional
import os
import mlflow
from mlflow.tracking import MlflowClient
from telecommunication.config import logger, PROJ_ROOT

# Initialize FastAPI app
app = FastAPI(title="Telecommunication Commission Prediction API")

# Initialize MLflow client
MLRUNS_DIR = PROJ_ROOT / "mlruns"
os.environ['MLFLOW_TRACKING_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
os.environ['MLFLOW_REGISTRY_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

client = MlflowClient()

class PredictionInput(BaseModel):
    bundle_type: int       # Changed to int
    operator: int          # Changed to int
    validity: float        # Already float
    regular_price: float   # Already float
    selling_price: float   # Already float
    internet: float        # Already float
    minutes: float         # Changed to float

class PredictionOutput(BaseModel):
    predicted_commission: float
    status: str
    error: Optional[str] = None

def get_production_model(model_name: str):
    """Get the production model"""
    try:
        # Use search_registered_models() instead of list_registered_models()
        registered_models = client.search_registered_models(filter_string=f"name='{model_name}'")
        logger.info(f"Available registered models: {[rm.name for rm in registered_models]}")
        
        # Get production versions
        versions = client.get_latest_versions(model_name, stages=["Production"])
        logger.info(f"Production versions found: {versions}")
        
        if versions:
            model_uri = f"models:/{model_name}/{versions[0].version}"
            logger.info(f"Loading model from URI: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)
        logger.error(f"No production version found for model: {model_name}")
        return None
    except Exception as e:
        logger.error(f"Error loading production model: {str(e)}")
        return None

@app.get("/")
async def root():
    return {"message": "Telecommunication Commission Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a single prediction"""
    try:
        # Add debug logging
        logger.debug(f"Received input data: {input_data}")
        
        # Convert input to DataFrame with exact column names matching training data
        df = pd.DataFrame([{
            'Bundle Type': input_data.bundle_type,
            'operator': input_data.operator,
            'validity': input_data.validity,
            'regularPrice': input_data.regular_price,
            'sellingPrice': input_data.selling_price,
            'Internet': input_data.internet,
            'Minutes': input_data.minutes
        }])
        
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        
        # Get production model - use the correct model name format
        model = get_production_model("RandomForest_model")  # or whichever model is in production
        
        if model is None:
            raise HTTPException(status_code=500, detail="No production model available")

        logger.debug("Model loaded successfully")
        
        # Make prediction
        try:
            prediction = model.predict(df)[0]
            logger.debug(f"Prediction made successfully: {prediction}")
            return PredictionOutput(
                predicted_commission=float(prediction),
                status="success"
            )
        except Exception as pred_error:
            logger.error(f"Prediction failed with error: {str(pred_error)}")
            raise Exception(f"Prediction error: {str(pred_error)}")

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
    "bundle_type": 1,              # int
    "operator": 2,                 # int
    "validity": 30.0,              # float
    "regular_price": 100.0,        # float
    "selling_price": 90.0,         # float
    "internet": 2.0,               # float
    "minutes": 100.0               # float
}

response = requests.post(url, json=data)
print(response.json())
"""