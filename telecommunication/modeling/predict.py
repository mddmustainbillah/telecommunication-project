import pandas as pd
from typing import Dict, Union
from telecommunication.config import logger
from telecommunication.modeling.registry import ModelRegistry

def predict(data: Union[pd.DataFrame, Dict]) -> Dict:
    """Make predictions using the production model"""
    try:
        # Convert dict to DataFrame if necessary
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Get production model
        registry = ModelRegistry()
        model = registry.get_production_model("best_model")
        
        if model is None:
            return {"error": "No production model available", "status": "failed"}

        # Make prediction
        predictions = model.predict(data)
        
        return {
            "predictions": predictions.tolist(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

def batch_predict(data_path: str) -> Dict:
    """Make predictions on a batch of data"""
    try:
        # Load data
        data = pd.read_csv(data_path)
        
        # Make predictions
        results = predict(data)
        
        if results["status"] == "success":
            # Save predictions
            predictions_df = pd.DataFrame({
                'prediction': results["predictions"]
            })
            predictions_df.to_csv(data_path.replace('.csv', '_predictions.csv'), index=False)
            
        return results

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }