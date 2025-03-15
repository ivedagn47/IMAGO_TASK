import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from src.preprocessing import load_dataset, run_data_exploration
from src.dimensionality_reduction import run_dimensionality_reduction
from src.model_training_evaluation import run_model_training_evaluation
from src.utils.logger import setup_logger
import joblib

logger = setup_logger('train')

def train():
    logger.info("Starting the training process")
    
    # Load and explore the dataset
    df = load_dataset()
    run_data_exploration()

    # Prepare data
    if 'vomitoxin_ppb' in df.columns:
        X = df.drop(['vomitoxin_ppb', 'hsi_id'], axis=1, errors='ignore')
        y = df['vomitoxin_ppb']
    else:
        X = df.iloc[:, :-1]  # Assuming all except the last column are features
        y = df.iloc[:, -1]   # Assuming the last column is the target
        logger.info("Assuming target variable is the last column.")

    # Feature scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    logger.debug("Features scaled successfully.")

    # Dimensionality reduction
    X_reduced, y_reduced = run_dimensionality_reduction(X_scaled, y, desired_components=10)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=0.2, random_state=42)
    logger.info("Data split into training and testing sets.")

    # Model training and evaluation
    results = run_model_training_evaluation(X_train, y_train, X_test, y_test)

    # Log results
    for result_name, result_content in results.items():
        logger.info(f"{result_name}: MAE = {result_content['mae']:.4f}, RMSE = {result_content['rmse']:.4f}, R² = {result_content['r2']:.4f}")

if __name__ == "__main__":
    train()
