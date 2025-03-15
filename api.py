from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import joblib
import os
import uvicorn
import tensorflow as tf
from src.preprocessing import load_dataset
from src.dimensionality_reduction import run_dimensionality_reduction
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('api')

app = FastAPI(title="Mycotoxin Concentration Prediction API",
              description="API for predicting DON concentration using hyperspectral data")

# Add CORS middleware to allow requests from any origin (important for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

def load_best_model_from_folder(folder="final"):
    """Load the best model found in the specified folder."""
    if not os.path.exists(folder):
        logger.error(f"Folder '{folder}' does not exist.")
        return None, None
        
    for file_name in os.listdir(folder):
        if file_name.endswith(".pkl"):
            model_path = os.path.join(folder, file_name)
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model, 'sklearn'
        elif file_name.endswith(".keras"):
            model_path = os.path.join(folder, file_name)
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model, 'keras'
    logger.error("No saved model found in the specified folder.")
    return None, None

def run_data_exploration_for_prediction(df):
    """Simplified version for prediction."""
    if 'hsi_id' in df.columns:
        X = df.drop(['hsi_id'], axis=1)
        sample_ids = df['hsi_id']
    else:
        X = df
        sample_ids = np.arange(len(X))
    
    # Ideally, load a saved scaler from training
    if os.path.exists('final/scaler.pkl'):
        scaler = joblib.load('final/scaler.pkl')
        X_scaled = scaler.transform(X)
    else:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled, sample_ids

def preprocess_input_data(df):
    """Apply existing preprocessing and dimensionality reduction."""
    X_scaled, sample_ids = run_data_exploration_for_prediction(df)
    X_reduced, _ = run_dimensionality_reduction(X_scaled, y=None, desired_components=10)
    return X_reduced, sample_ids

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model, model_type
    model, model_type = load_best_model_from_folder("final")
    if model is None:
        logger.error("Failed to load model during startup")
    else:
        logger.info(f"Successfully loaded {model_type} model during startup")

@app.get("/")
async def root():
    return {"message": "Welcome to the Mycotoxin Concentration Prediction API"}

@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with hyperspectral data to get DON concentration predictions.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read the CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        logger.info(f"Successfully read CSV with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Preprocess the data
        X_reduced, sample_ids = preprocess_input_data(df)
        logger.info(f"Data preprocessed successfully, reduced to {X_reduced.shape[1]} features")
        
        # Make predictions
        if model_type == 'sklearn':
            predictions = model.predict(X_reduced)
        elif model_type == 'keras':
            predictions = model.predict(X_reduced).flatten()
        else:
            raise HTTPException(status_code=500, detail="Unknown model type or model not loaded")
        
        # Prepare results
        results = {
            "success": True,
            "predictions": predictions.tolist(),
            "sample_ids": sample_ids.tolist() if isinstance(sample_ids, np.ndarray) else sample_ids,
            "message": "DON concentration prediction successful"
        }
        
        logger.info(f"Successfully generated predictions for {len(predictions)} samples")
        return results
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)