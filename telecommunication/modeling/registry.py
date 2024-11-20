import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict
from telecommunication.config import logger, PROJ_ROOT

# Initialize MLflow client and set tracking URI
MLRUNS_DIR = PROJ_ROOT / "mlruns"
os.environ['MLFLOW_TRACKING_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
os.environ['MLFLOW_REGISTRY_URI'] = f"file://{str(MLRUNS_DIR.absolute())}"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()

    def get_production_model(self, model_name: str):
        """Get the production model"""
        try:
            logger.info(f"Attempting to load production model: {model_name}")
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            
            # Search for registered models
            registered_models = self.client.search_registered_models(filter_string=f"name='{model_name}'")
            logger.info(f"Found registered models: {[rm.name for rm in registered_models]}")
            
            if not registered_models:
                logger.error(f"No registered model found with name: {model_name}")
                return None
            
            # Get production versions
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            logger.info(f"Production versions found: {versions}")
            
            if versions:
                version = versions[0]
                # Get the run ID and load model directly from the run artifacts
                run_id = version.run_id
                experiment_id = self.client.get_run(run_id).info.experiment_id
                artifact_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
                
                logger.info(f"Loading model from artifact URI: {artifact_uri}")
                model = mlflow.sklearn.load_model(artifact_uri)
                return model
            
            logger.error(f"No production version found for model: {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}", exc_info=True)
            return None

    def promote_staging_to_production(self, model_name: str) -> bool:
        """Promote staging model to production"""
        try:
            # Get staging model
            staging_model = self.get_staging_model(model_name)
            if not staging_model:
                logger.warning(f"No staging model found for {model_name}")
                return False

            # Get current production versions and archive them
            production_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            for version in production_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                logger.info(f"Archived model {model_name} version {version.version}")

            # Transition staging to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=staging_model['version'],
                stage="Production"
            )
            logger.info(f"Model {model_name} version {staging_model['version']} promoted to production")
            return True

        except Exception as e:
            logger.error(f"Error promoting model to production: {str(e)}")
            return False

    def get_staging_model(self, model_name: str) -> Optional[Dict]:
        """Get current staging model details"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Staging"])
            if versions:
                version = versions[0]
                return {
                    'version': version.version,
                    'run_id': version.run_id,
                    'current_stage': version.current_stage
                }
            return None
        except Exception as e:
            logger.error(f"Error getting staging model: {str(e)}")
            return None

def main():
    """Promote the best model from staging to production"""
    try:
        registry = ModelRegistry()
        # Get all registered models
        registered_models = registry.client.search_registered_models()
        
        for rm in registered_models:
            model_name = rm.name
            logger.info(f"Processing model: {model_name}")
            
            # If there's a staging model, promote it
            success = registry.promote_staging_to_production(model_name)
            if success:
                logger.info(f"Successfully promoted {model_name} to production")
            else:
                logger.warning(f"Failed to promote {model_name} to production")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()