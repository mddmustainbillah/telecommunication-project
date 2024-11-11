from datetime import datetime, timedelta
import mlflow
from telecommunication.config import logger
from telecommunication.modeling.train import main as train_models
from telecommunication.modeling.registry import ModelRegistry

def should_retrain(performance_threshold: float = 0.5, 
                  days_threshold: int = 30) -> bool:
    """Determine if models should be retrained"""
    try:
        registry = ModelRegistry()
        client = mlflow.tracking.MlflowClient()
        
        # Get current production model
        production_versions = client.get_latest_versions("best_model", stages=["Production"])
        if not production_versions:
            return True
            
        # Check performance
        metrics = registry.get_model_metrics("best_model")
        current_performance = metrics[production_versions[0].version]['metrics'].get('r2', 0)
        if current_performance < performance_threshold:
            return True
            
        # Check age
        model_date = datetime.fromtimestamp(production_versions[0].creation_timestamp/1000)
        if datetime.now() - model_date > timedelta(days=days_threshold):
            return True
            
        return False

    except Exception as e:
        logger.error(f"Error checking retrain criteria: {str(e)}")
        return True

def retrain():
    """Retrain models if needed"""
    try:
        if should_retrain():
            logger.info("Starting model retraining...")
            train_models()
            logger.info("Retraining completed successfully")
            return True
        else:
            logger.info("Retraining not needed at this time")
            return False

    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        return False

if __name__ == "__main__":
    retrain()