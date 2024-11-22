from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from telecommunication.dataset import fetch_telco_data, save_data
from telecommunication.features import (
    clean_data, 
    feature_engineering, 
    feature_mapping, 
    split_and_save_data
)
from telecommunication.modeling.train import (
    load_data,
    setup_mlflow_experiment,
    train_and_evaluate_model
)
from telecommunication.modeling.registry import ModelRegistry
from telecommunication.config import (
    logger,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR
)
from utils import load_params

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def fetch_and_save_raw_data():
    """Fetch data from SQLite and save raw data"""
    try:
        logger.info("Fetching raw data...")
        df = fetch_telco_data()
        
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_data(df, RAW_DATA_DIR / "raw_dataset.csv")
        logger.info("Raw data saved successfully")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_and_save_raw_data: {str(e)}")
        raise

@task
def process_features(df: pd.DataFrame):
    """Process and engineer features"""
    try:
        logger.info("Processing features...")
        
        # Clean data
        logger.info("Cleaning data...")
        df_cleaned = clean_data(df)
        
        # Engineer features
        logger.info("Engineering features...")
        df_engineered = feature_engineering(df_cleaned)
        
        # Map features
        logger.info("Mapping features...")
        df_mapped = feature_mapping(df_engineered)
        
        # Save interim data
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_mapped.to_csv(INTERIM_DATA_DIR / "interim_dataset.csv", index=False)
        logger.info("Processed features saved to interim directory")
        
        return df_mapped
    except Exception as e:
        logger.error(f"Error in process_features: {str(e)}")
        raise

@task
def split_data(df: pd.DataFrame):
    """Split data into train and test sets"""
    try:
        logger.info("Splitting data...")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        train_df, test_df = split_and_save_data(df)
        logger.info("Data split and saved successfully")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error in split_data: {str(e)}")
        raise

@task
def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train and evaluate models"""
    try:
        logger.info("Training models...")
        
        # Load parameters
        params = load_params()
        target_column = params['training']['target_column']
        
        # Setup MLflow experiment
        experiment_id = setup_mlflow_experiment()
        logger.info(f"MLflow experiment ID: {experiment_id}")
        
        # Prepare data
        X_train, y_train, X_test, y_test = load_data(
            PROCESSED_DATA_DIR / "train_dataset.csv",
            PROCESSED_DATA_DIR / "test_dataset.csv",
            target_column
        )
        
        # Train models
        results = {}
        for model_name, model_config in params['models'].items():
            try:
                logger.info(f"Training {model_name}...")
                model, metrics, run_id = train_and_evaluate_model(
                    model_name,
                    model_config,
                    X_train, y_train,
                    X_test, y_test,
                    MODELS_DIR,
                    REPORTS_DIR / "metrics",
                    params['training']
                )
                results[model_name] = (model, metrics, run_id)
                logger.info(f"{model_name} training completed. Run ID: {run_id}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        raise

@task
def promote_models():
    """Promote models to production"""
    try:
        logger.info("Promoting models...")
        registry = ModelRegistry()
        registered_models = registry.client.search_registered_models()
        
        for rm in registered_models:
            model_name = rm.name
            logger.info(f"Processing model: {model_name}")
            
            success = registry.promote_staging_to_production(model_name)
            if success:
                logger.info(f"Successfully promoted {model_name} to production")
            else:
                logger.warning(f"Failed to promote {model_name} to production")
    except Exception as e:
        logger.error(f"Error in promote_models: {str(e)}")
        raise

@flow(name="Telco Commission Prediction Pipeline")
def main_pipeline():
    """Main pipeline flow"""
    try:
        # Fetch and save raw data
        df = fetch_and_save_raw_data()
        
        # Process features
        df_processed = process_features(df)
        
        # Split data
        train_df, test_df = split_data(df_processed)
        
        # Train models
        results = train_models(train_df, test_df)
        
        # Promote models
        promote_models()
        
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main_pipeline()