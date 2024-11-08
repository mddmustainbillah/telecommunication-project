# import logging
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import Tuple, List, Dict

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import cross_val_score, GridSearchCV
# import joblib
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt

# from utils import load_params
# from telecommunication.config import logger

# def load_data(train_path: Path, test_path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
#     """Load and split the train and test datasets into features and labels"""
#     try:
#         # Load train and test data
#         train_df = pd.read_csv(train_path)
#         test_df = pd.read_csv(test_path)
        
#         # Separate features and target
#         X_train = train_df.drop(columns=[target_column])
#         y_train = train_df[target_column]
#         X_test = test_df.drop(columns=[target_column])
#         y_test = test_df[target_column]
        
#         return X_train, y_train, X_test, y_test
#     except Exception as e:
#         logger.error(f"Error loading data: {str(e)}")
#         raise

# def perform_cross_validation(
#     model: RandomForestRegressor,
#     X: pd.DataFrame,
#     y: pd.Series,
#     cv: int = 5
# ) -> Dict[str, float]:
#     """Perform cross-validation and return scores"""
#     cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
#     cv_rmse_scores = np.sqrt(-cross_val_score(
#         model, X, y, cv=cv, scoring='neg_mean_squared_error'
#     ))
    
#     return {
#         'cv_r2_mean': cv_scores.mean(),
#         'cv_r2_std': cv_scores.std(),
#         'cv_rmse_mean': cv_rmse_scores.mean(),
#         'cv_rmse_std': cv_rmse_scores.std()
#     }

# def tune_hyperparameters(
#     X: pd.DataFrame,
#     y: pd.Series,
#     params: Dict
# ) -> Tuple[RandomForestRegressor, Dict]:
#     """Perform hyperparameter tuning using GridSearchCV"""
#     param_grid = {
#         'n_estimators': params['model']['n_estimators_grid'],
#         'max_depth': params['model']['max_depth_grid'],
#         'min_samples_split': params['model']['min_samples_split_grid'],
#         'min_samples_leaf': params['model']['min_samples_leaf_grid']
#     }
    
#     base_model = RandomForestRegressor(random_state=params['model']['random_state'])
#     grid_search = GridSearchCV(
#         estimator=base_model,
#         param_grid=param_grid,
#         cv=5,
#         scoring='r2',
#         n_jobs=-1
#     )
    
#     grid_search.fit(X, y)
    
#     return grid_search.best_estimator_, grid_search.best_params_

# def evaluate_model(
#     model: RandomForestRegressor,
#     X_test: pd.DataFrame,
#     y_test: pd.Series
# ) -> Dict[str, float]:
#     """Evaluate the model and return metrics"""
#     predictions = model.predict(X_test)
    
#     metrics = {
#         'mse': mean_squared_error(y_test, predictions),
#         'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
#         'mae': mean_absolute_error(y_test, predictions),
#         'r2': r2_score(y_test, predictions)
#     }
    
#     return metrics

# def log_feature_importance(
#     model: RandomForestRegressor,
#     features: List[str]
# ) -> pd.DataFrame:
#     """Get feature importance and log to MLflow"""
#     feature_importance = pd.DataFrame({
#         'feature': features,
#         'importance': model.feature_importances_
#     }).sort_values('importance', ascending=False)
    
#     # Create and save feature importance plot
#     plt.figure(figsize=(10, 6))
#     plt.bar(feature_importance['feature'], feature_importance['importance'])
#     plt.xticks(rotation=45)
#     plt.title('Feature Importance')
#     plt.tight_layout()
    
#     # Log plot to MLflow
#     mlflow.log_figure(plt.gcf(), "feature_importance.png")
#     plt.close()
    
#     # Log feature importance as a CSV
#     temp_path = Path("feature_importance.csv")
#     feature_importance.to_csv(temp_path, index=False)
#     mlflow.log_artifact(temp_path)
#     temp_path.unlink()  # Clean up temporary file
    
#     return feature_importance

# def save_model_artifacts(
#     model: RandomForestRegressor,
#     metrics: Dict[str, float],
#     feature_importance: pd.DataFrame,
#     output_dir: Path
# ) -> None:
#     """Save model and artifacts locally"""
#     output_dir.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, output_dir / 'model.joblib')
#     pd.DataFrame([metrics]).to_csv(output_dir / 'metrics.csv', index=False)
#     feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)

# def main(
#     train_path: Path = Path("data/processed/train.csv"),
#     test_path: Path = Path("data/processed/test.csv"),
#     model_dir: Path = Path("models"),
#     target_column: str = "target"  # Replace with your actual target column name
# ):
#     """Main training pipeline"""
#     try:
#         # Setup logging
#         logging.basicConfig(level=logging.INFO)
#         logger.info("Starting training pipeline...")
        
#         # Load parameters
#         params = load_params()
        
#         # Set MLflow tracking URI if needed
#         # mlflow.set_tracking_uri("your_tracking_uri")
        
#         # Start MLflow run
#         with mlflow.start_run():
#             # Log the parameters file
#             mlflow.log_artifact("params.yaml")
            
#             # Load data
#             logger.info("Loading training and test data...")
#             X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)
            
#             # Log dataset info
#             mlflow.log_param("n_samples", len(X_train))
#             mlflow.log_param("n_features", X_train.shape[1])
            
#             # Perform hyperparameter tuning
#             logger.info("Tuning hyperparameters...")
#             best_model, best_params = tune_hyperparameters(X_train, y_train, params)
#             mlflow.log_params(best_params)
            
#             # Perform cross-validation
#             logger.info("Performing cross-validation...")
#             cv_metrics = perform_cross_validation(best_model, X_train, y_train)
#             mlflow.log_metrics(cv_metrics)
            
#             # Train final model
#             logger.info("Training final model...")
#             best_model.fit(X_train, y_train)
            
#             # Evaluate model
#             logger.info("Evaluating model...")
#             metrics = evaluate_model(best_model, X_train, y_train)
#             mlflow.log_metrics(metrics)
            
#             # Log feature importance
#             feature_importance = log_feature_importance(
#                 best_model, X_train.columns.tolist()
#             )
            
#             # Log model to MLflow
#             mlflow.sklearn.log_model(
#                 best_model, 
#                 "model",
#                 registered_model_name="RandomForestRegressor"
#             )
            
#             # Save artifacts locally
#             logger.info("Saving model artifacts...")
#             save_model_artifacts(best_model, metrics, feature_importance, model_dir)
            
#             logger.info("Training pipeline completed successfully!")
            
#     except Exception as e:
#         logger.error(f"Error in training pipeline: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from utils import load_params

from telecommunication.config import logger, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR


def load_data(train_path: Path, test_path: Path, target_column: str):
    """ Load and split the train and test datasets into features and labels """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model and return metrics """
    predictions = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, predictions),     
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }
    return metrics

def save_model_artifacts(model, metrics, output_dir: Path, metrics_dir: Path):
    """ Save model and artifacts locally"""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f'{model}.joblib')
    pd.DataFrame([metrics]).to_csv(metrics_dir / 'metrics.csv', index=False)



def main():
    """ Main training pipeline """
    train_path = PROCESSED_DATA_DIR / "train_dataset.csv"
    test_path = PROCESSED_DATA_DIR / "test_dataset.csv" 
    model_dir = MODELS_DIR
    metrics_dir = REPORTS_DIR / "metrics"
    target_column = "commission"

    # Load parameters
    print("Loading parameters...")
    params = load_params()

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, target_column)

    # Train model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=params['model']['n_estimators'],
        random_state=params['model']['random_state']
    )   
    model.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

    # Save model and artifacts
    print("Saving model and artifacts...")
    save_model_artifacts(model, metrics, model_dir, metrics_dir)

    print("Training pipeline completed successfully!")



if __name__ == "__main__":
    main()  
