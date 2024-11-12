import pandas as pd
from typing import Dict, Union
from telecommunication.config import logger
from telecommunication.modeling.registry import ModelRegistry

# Mapping dictionaries
bundle_type_mapping = {
    "Internet & Minute": 1,
    "Internet": 2,
    "Minute": 3
}

operator_mapping = {
    "robi": 1,
    "gp": 2,
    "airtel": 3,
    "bl": 4
}

def map_input_data(input_data: Dict[str, Union[str, float]]) -> Dict[str, Union[int, float]]:
    """Map input data from string to the required types for prediction."""
    try:
        bundle_type = bundle_type_mapping.get(input_data['bundle_type'])
        operator = operator_mapping.get(input_data['operator'])

        if bundle_type is None or operator is None:
            raise ValueError("Invalid bundle type or operator")

        mapped_data = {
            'Bundle Type': bundle_type,
            'operator': operator,
            'validity': input_data['validity'],
            'regularPrice': input_data['regular_price'],
            'sellingPrice': input_data['selling_price'],
            'Internet': input_data['internet'],
            'Minutes': input_data['minutes']
        }
        return mapped_data

    except Exception as e:
        logger.error(f"Error mapping input data: {str(e)}")
        raise

def predict(data: Dict[str, Union[str, float]]) -> Dict:
    """Make predictions using the production model."""
    try:
        # Map input data
        mapped_data = map_input_data(data)

        # Convert mapped data to DataFrame
        df = pd.DataFrame([mapped_data])

        # Get production model
        registry = ModelRegistry()
        model = registry.get_production_model("RandomForest_model")  # or whichever model is in production
        
        if model is None:
            return {"error": "No production model available", "status": "failed"}

        # Make prediction
        prediction = model.predict(df)[0]
        return {
            "predictions": float(prediction),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }