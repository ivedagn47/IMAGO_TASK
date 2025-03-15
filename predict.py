import pandas as pd
import joblib
from src.utils.logger import setup_logger

logger = setup_logger('predict')

def load_model(path):
    logger.info(f"Loading model from {path}")
    return joblib.load(path)

def predict(data):
    model = load_model('final/best_model.pkl')
    scaler = joblib.load('final/scaler.pkl')
    logger.debug("Loaded model and scaler successfully.")

    X_scaled = scaler.transform(data)
    predictions = model.predict(X_scaled)
    logger.info("Prediction made successfully.")
    return predictions

if _name_ == "_main_"
