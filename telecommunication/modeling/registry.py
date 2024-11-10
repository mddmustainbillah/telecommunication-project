import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict
from datetime import datetime
from telecommunication.config import logger

class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()
        self.performance_threshold = 0.5  # Minimum R² score required for production

    def get_model_version(self, model_name: str, stage: str = None) -> Optional[Dict]:
        """Get latest version of model in specified stage"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage] if stage else None)
            if not versions:
                return None
            return {
                'version': versions[0].version,
                'stage': versions[0].current_stage,
                'run_id': versions[0].run_id
            }
        except Exception as e:
            logger.error(f"Error getting model version: {str(e)}")
            return None

    def promote_to_production(self, model_name: str) -> bool:
        """Promote staging model to production if it meets criteria"""
        try:
            # Get staging model
            staging_model = self.get_model_version(model_name, "Staging")
            if not staging_model:
                logger.warning(f"No {model_name} model in staging")
                return False

            # Get model metrics
            run = self.client.get_run(staging_model['run_id'])
            r2_score = run.data.metrics.get('r2', 0)

            # Check if model meets performance threshold
            if r2_score > self.performance_threshold:
                # Archive current production model if exists
                current_prod = self.get_model_version(model_name, "Production")
                if current_prod:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=current_prod['version'],
                        stage="Archived"
                    )

                # Promote staging to production
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=staging_model['version'],
                    stage="Production"
                )
                logger.info(f"Model {model_name} v{staging_model['version']} promoted to production")
                return True
            
            logger.warning(f"Model performance {r2_score} below threshold {self.performance_threshold}")
            return False

        except Exception as e:
            logger.error(f"Error promoting model to production: {str(e)}")
            return False

    def get_model_info(self, model_name: str) -> Dict:
        """Get information about all versions of a model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return {
                'model_name': model_name,
                'versions': [{
                    'version': v.version,
                    'stage': v.current_stage,
                    'run_id': v.run_id,
                    'metrics': self.client.get_run(v.run_id).data.metrics,
                    'timestamp': datetime.fromtimestamp(v.creation_timestamp/1000)
                } for v in versions]
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
