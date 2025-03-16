'''
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import joblib
import os
import uvicorn
import tensorflow as tf

# Import your already created modules
from src.preprocessing import load_dataset  # if needed elsewhere
from src.dimensionality_reduction import run_dimensionality_reduction
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('api')

app = FastAPI(
    title="Mycotoxin Concentration Prediction API",
    description="API for predicting DON concentration using hyperspectral data"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """
    Simplified data exploration for prediction.
    Drops 'hsi_id' if present, scales data, and returns sample IDs.
    """
    if 'hsi_id' in df.columns:
        X = df.drop(['hsi_id'], axis=1)
        sample_ids = df['hsi_id']
    else:
        X = df
        sample_ids = np.arange(len(X))
    
    # Load a saved scaler if available; otherwise fit a new RobustScaler
    scaler_path = os.path.join("final", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        logger.info("Loaded saved scaler.")
    else:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("No saved scaler found; fitted new RobustScaler.")
    
    return X_scaled, sample_ids

def preprocess_input_data(df):
    """
    Applies data preprocessing and dimensionality reduction using your modules.
    The 'skip_plots=True' flag is passed so that no plots are generated during prediction.
    """
    X_scaled, sample_ids = run_data_exploration_for_prediction(df)
    X_reduced, _ = run_dimensionality_reduction(X_scaled, y=None, desired_components=10, skip_plots=True)
    return X_reduced, sample_ids

# Load model at startup using your module
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
        
        # Preprocess data using your existing modules
        X_reduced, sample_ids = preprocess_input_data(df)
        logger.info(f"Preprocessed data successfully; reduced to {X_reduced.shape[1]} features")
        
        # Make predictions based on model type
        if model_type == 'sklearn':
            predictions = model.predict(X_reduced)
        elif model_type == 'keras':
            predictions = model.predict(X_reduced).flatten()
        else:
            raise HTTPException(status_code=500, detail="Unknown model type or model not loaded")
        
        results = {
            "success": True,
            "predictions": predictions.tolist(),
            "sample_ids": sample_ids.tolist() if isinstance(sample_ids, np.ndarray) else sample_ids,
            "message": "DON concentration prediction successful"
        }
        
        logger.info(f"Generated predictions for {len(predictions)} samples")
        return results
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
'''
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import joblib
import os
import uvicorn
import tensorflow as tf

# Import your existing modules exactly as defined
from src.preprocessing import load_dataset  # if needed elsewhere
from src.dimensionality_reduction import run_dimensionality_reduction
from src.utils.logger import setup_logger

logger = setup_logger('api')

app = FastAPI(
    title="Mycotoxin Concentration Prediction API",
    description="API for predicting DON concentration using hyperspectral data"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_best_model_from_folder(folder="final"):
    """
    Load the best model found in the specified folder.
    Ensures that only valid prediction models are loaded, not scalers or other objects.
    """
    if not os.path.exists(folder):
        logger.error(f"Folder '{folder}' does not exist.")
        return None, None
    
    # First try to find files with 'model' in the name
    model_candidates = [f for f in os.listdir(folder) 
                       if ('model' in f.lower() or 'classifier' in f.lower() or 'regressor' in f.lower())
                       and (f.endswith('.pkl') or f.endswith('.keras'))]
    
    # If no explicit model files found, try all .pkl and .keras files except known non-models
    if not model_candidates:
        model_candidates = [f for f in os.listdir(folder) 
                           if (f.endswith('.pkl') or f.endswith('.keras'))
                           and not any(x in f.lower() for x in ['scaler', 'encoder', 'transformer', 'pca'])]
    
    # Try loading each candidate and check if it has a predict method
    for file_name in model_candidates:
        model_path = os.path.join(folder, file_name)
        try:
            if file_name.endswith('.pkl'):
                model = joblib.load(model_path)
                if hasattr(model, 'predict') and callable(model.predict):
                    logger.info(f"Loaded sklearn model from {model_path}")
                    return model, 'sklearn'
                else:
                    logger.warning(f"File {file_name} does not contain a valid model with predict method")
            elif file_name.endswith('.keras'):
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded keras model from {model_path}")
                return model, 'keras'
        except Exception as e:
            logger.error(f"Error loading {file_name}: {str(e)}")
            continue
    
    # If we get here, no valid model was found
    logger.error("No valid model with 'predict' method found in the folder.")
    return None, None

def run_data_exploration_for_prediction(df):
    """
    Simplified data exploration for prediction.
    Drops the 'hsi_id' column if present, scales the data,
    and returns both the scaled data and sample IDs.
    """
    if 'hsi_id' in df.columns:
        X = df.drop(['hsi_id'], axis=1)
        sample_ids = df['hsi_id']
    else:
        X = df
        sample_ids = np.arange(len(X))
    
    scaler_path = os.path.join("final", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        logger.info("Loaded saved scaler.")
    else:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("No saved scaler found; fitted new RobustScaler.")
    
    return X_scaled, sample_ids

def preprocess_input_data(df):
    """
    Applies data preprocessing and dimensionality reduction
    using your existing modules. Note that we call the functions
    exactly as defined in your modules.
    """
    X_scaled, sample_ids = run_data_exploration_for_prediction(df)
    X_reduced, _ = run_dimensionality_reduction(X_scaled, y=None, desired_components=10)
    return X_reduced, sample_ids

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
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        logger.info(f"Successfully read CSV with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Preprocess the data using your existing functions
        X_reduced, sample_ids = preprocess_input_data(df)
        logger.info(f"Preprocessed data successfully; reduced to {X_reduced.shape[1]} features")
        
        # Make predictions using the loaded model
        if model_type == 'sklearn':
            predictions = model.predict(X_reduced)
        elif model_type == 'keras':
            predictions = model.predict(X_reduced).flatten()
        else:
            raise HTTPException(status_code=500, detail="Unknown model type or model not loaded")
        
        results = {
            "success": True,
            "predictions": predictions.tolist(),
            "sample_ids": sample_ids.tolist() if isinstance(sample_ids, np.ndarray) else sample_ids,
            "message": "DON concentration prediction successful"
        }
        
        logger.info(f"Generated predictions for {len(predictions)} samples")
        return results
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
