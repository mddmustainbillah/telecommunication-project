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
        # Add logging for debugging
        logger.info(f"Starting prediction with data: {data}")
        
        # Map input data
        mapped_data = map_input_data(data)
        logger.info(f"Mapped data: {mapped_data}")

        # Convert mapped data to DataFrame
        df = pd.DataFrame([mapped_data])
        logger.info(f"Created DataFrame: {df}")

        # Get production model
        registry = ModelRegistry()
        model = registry.get_production_model("RandomForest_model")
        
        if model is None:
            logger.error("No production model available")
            return {"error": "No production model available", "status": "failed"}

        # Make prediction
        prediction = model.predict(df)[0]
        logger.info(f"Made prediction: {prediction}")
        
        return {
            "predictions": float(prediction),
            "status": "success"
        }

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return {
            "error": str(ve),
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)  # Add full traceback
        return {
            "error": str(e),
            "status": "failed"
        }