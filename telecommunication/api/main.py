from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import mlflow
from mlflow.tracking import MlflowClient
from telecommunication.config import logger, PROJ_ROOT
from telecommunication.modeling.predict import predict
from typing import Optional
from pathlib import Path

# Get the current directory
BASE_PATH = Path(__file__).resolve().parent

# Initialize FastAPI app
app = FastAPI(title="Telecommunication Commission Prediction API")

# Configure static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_PATH / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_PATH / "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MLflow client
MLRUNS_DIR = PROJ_ROOT / "mlruns"
os.environ['MLFLOW_TRACKING_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
os.environ['MLFLOW_REGISTRY_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

client = MlflowClient()

class PredictionInput(BaseModel):
    bundle_type: str
    operator: str
    validity: float
    regular_price: float
    selling_price: float
    internet: float
    minutes: float

class PredictionOutput(BaseModel):
    predicted_commission: float
    status: str
    error: Optional[str] = None

@app.get("/")
async def home(request: Request):
    """Serve the index.html template"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    """Make a single prediction"""
    try:
        input_data_dict = input_data.dict()
        logger.info(f"Received input data: {input_data_dict}")
        
        result = predict(input_data_dict)
        logger.info(f"Prediction result: {result}")

        if result["status"] == "success":
            return PredictionOutput(
                predicted_commission=result["predictions"],
                status="success"
            )
        else:
            return PredictionOutput(
                predicted_commission=0.0,
                status="failed",
                error=result.get("error", "Unknown error occurred")
            )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return PredictionOutput(
            predicted_commission=0.0,
            status="failed",
            error=str(e)
        )