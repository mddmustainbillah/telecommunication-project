import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
import warnings
import json
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from telecommunication.config import (
    logger, 
    PROCESSED_DATA_DIR, 
    MODELS_DIR, 
    REPORTS_DIR,
    PROJ_ROOT
)
from utils import load_params

# Suppress warnings
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")

# Set absolute path for MLflow tracking URI
mlflow_path = str(PROJ_ROOT.absolute() / 'mlruns')
os.environ['MLFLOW_TRACKING_URI'] = f"file://{mlflow_path}"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Ensure we're working from project root
os.chdir(str(PROJ_ROOT))

# Ensure directories exist
PROJ_ROOT.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR = PROJ_ROOT / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJ_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(train_path: Path, test_path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and split the train and test datasets into features and labels"""
    logger.info("Loading training and test data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return metrics"""
    predictions = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, predictions),     
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }
    return metrics

def save_model_artifacts(model: Any, metrics: Dict[str, float], 
                        output_dir: Path, metrics_dir: Path, 
                        is_best_model: bool = False) -> None:
    """Save model and artifacts locally"""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    if is_best_model:
        model_path = output_dir / 'best_model.joblib'
        metrics_path = metrics_dir / 'best_metrics.json'
    else:
        model_path = output_dir / 'model.joblib'
        metrics_path = metrics_dir / 'metrics.json'
        
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = metrics_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Log artifacts to MLflow
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(metrics_path)

def get_model_and_params(model_name: str, model_config: Dict) -> Tuple[Any, Dict]:
    """Get model instance and parameter grid based on model type"""
    model_type = model_config['model_type']
    random_state = model_config.get('random_state', 42)
    
    if model_type == "RandomForest":
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': model_config['params']['n_estimators_grid'],
            'max_depth': model_config['params']['max_depth_grid'],
            'min_samples_split': model_config['params']['min_samples_split_grid'],
            'min_samples_leaf': model_config['params']['min_samples_leaf_grid']
        }
    
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': model_config['params']['n_estimators_grid'],
            'learning_rate': model_config['params']['learning_rate_grid'],
            'max_depth': model_config['params']['max_depth_grid'],
            'min_samples_split': model_config['params']['min_samples_split_grid']
        }
    
    elif model_type == "LinearRegression":
        model = LinearRegression()
        param_grid = {
            'fit_intercept': model_config['params']['fit_intercept_grid'],
            'positive': model_config['params']['positive_grid']
        }
    
    elif model_type == "DecisionTree":
        model = DecisionTreeRegressor(random_state=random_state)
        param_grid = {
            'max_depth': model_config['params']['max_depth_grid'],
            'min_samples_split': model_config['params']['min_samples_split_grid'],
            'min_samples_leaf': model_config['params']['min_samples_leaf_grid'],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, param_grid

def tune_hyperparameters(model: Any, param_grid: Dict, X: pd.DataFrame, y: pd.Series, 
                        model_name: str, n_iter: int, cv: int) -> Tuple[Any, Dict, Dict]:
    """Perform hyperparameter tuning using RandomizedSearchCV or GridSearchCV"""
    # Calculate total parameter combinations
    n_combinations = 1
    for param_values in param_grid.values():
        n_combinations *= len(param_values)
    
    # Use GridSearchCV for small parameter spaces
    if n_combinations <= n_iter:
        logger.info(f"Using GridSearchCV for {model_name}")
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
    else:
        logger.info(f"Using RandomizedSearchCV for {model_name}")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    logger.info(f"Starting parameter search for {model_name}...")
    search.fit(X, y)
    logger.info(f"Best score: {search.best_score_}")
    logger.info(f"Best parameters: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.cv_results_

def train_and_evaluate_model(model_name: str, model_config: Dict, 
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series, 
                           model_dir: Path, metrics_dir: Path, 
                           training_params: Dict) -> Tuple[Any, Dict, str]:
    """Train and evaluate a single model"""
    # Create model-specific directories
    model_metrics_dir = metrics_dir / model_name
    model_metrics_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                         nested=True) as run:
        # Get model and parameter grid
        model, param_grid = get_model_and_params(model_name, model_config)
        
        # Log model configuration
        mlflow.log_param("model_type", model_config['model_type'])
        mlflow.log_params({f"config_{k}": str(v) for k, v in model_config.items() 
                          if k != 'params'})
        
        # Tune hyperparameters
        logger.info(f"Training {model_name}...")
        best_model, best_params, cv_results = tune_hyperparameters(
            model, param_grid, X_train, y_train, model_name,
            training_params['n_iter_search'], training_params['cv_folds']
        )

        # Log parameters and CV results
        mlflow.log_params(best_params)
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_results.items() 
                           if isinstance(v, (int, float))})

        # Save CV results in model-specific directory
        cv_results_path = model_metrics_dir / f"{model_name}_cv_results.json"
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, default=lambda x: x.tolist() 
                     if isinstance(x, np.ndarray) else x)
        mlflow.log_artifact(cv_results_path)

        # Evaluate on test set
        test_metrics = evaluate_model(best_model, X_test, y_test)
        logger.info(f"{model_name} test metrics: {test_metrics}")
        mlflow.log_metrics(test_metrics)

        # Log feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = model_metrics_dir / f"{model_name}_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

        # Register model
        model_uri = mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name=f"{model_name}_model"
        ).model_uri

        # Model version tracking 
        client = MlflowClient()
        latest_version = client.get_latest_versions(f"{model_name}_model")[0].version
        model_version = client.get_model_version(
            name=f"{model_name}_model",
            version=latest_version
        )
        mlflow.log_param("model_version", model_version.version)

        # Save artifacts locally
        save_model_artifacts(best_model, test_metrics, 
                           model_dir / model_name,  # Model directory
                           model_metrics_dir)       # Model-specific metrics directory

        return best_model, test_metrics, run.info.run_id
    
def setup_mlflow_experiment():
    """Set up MLflow experiment"""
    experiment_name = "telco_commission_prediction"
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"version": "1.0", "priority": "high"}
        )
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def main():
    """Main training pipeline"""
    # Load parameters
    params = load_params()
    
    # Setup paths
    train_path = PROCESSED_DATA_DIR / "train_dataset.csv"
    test_path = PROCESSED_DATA_DIR / "test_dataset.csv" 
    model_dir = MODELS_DIR
    metrics_dir = REPORTS_DIR / "metrics"
    target_column = params['training']['target_column']

    # Setup MLflow experiment
    experiment_id = setup_mlflow_experiment()

    # Start MLflow run
    run_name = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        mlflow.set_tags({
            "pipeline_version": "1.0",
            "data_version": datetime.now().strftime("%Y%m%d"),
            "environment": "development"
        })
        # Load data
        X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

        # Log dataset info
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("features", list(X_train.columns))

        # Train and evaluate each model
        results = {}
        for model_name, model_config in params['models'].items():
            try:
                model, metrics, run_id = train_and_evaluate_model(
                    model_name,
                    model_config,
                    X_train, y_train,
                    X_test, y_test,
                    model_dir,
                    metrics_dir,
                    params['training']
                )
                results[model_name] = (model, metrics, run_id)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue

        # Create and log comparison DataFrame
        comparison_df = pd.DataFrame({name: metrics for name, (_, metrics, _) 
                                    in results.items()}).T
        comparison_path = metrics_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path)
        mlflow.log_artifact(comparison_path)

        # Find best model
        best_model_name = comparison_df['r2'].idxmax()
        best_model = results[best_model_name][0]  # Get the actual model object
        best_model_metrics = results[best_model_name][1]
        best_run_id = results[best_model_name][2]

        # Save best model separately
        save_model_artifacts(
            model=best_model,
            metrics=best_model_metrics,
            output_dir=model_dir,  # Save directly in models directory
            metrics_dir=metrics_dir,
            is_best_model=True  # This will save it as 'best_model.joblib'
        )

        # Log best model information
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_metrics({f"best_{k}": v for k, v in best_model_metrics.items()})

        # Transition best model to staging
        client = MlflowClient()
        model_version = client.get_latest_versions(f"{best_model_name}_model")[0]
        client.transition_model_version_stage(
            name=f"{best_model_name}_model",
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"\nBest model: {best_model_name}")
        logger.info("\nModel comparison:")
        logger.info(comparison_df)

    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()