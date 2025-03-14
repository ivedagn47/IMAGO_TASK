import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Import your custom modules
from src.preprocessing import load_dataset, run_data_exploration
from src.dimensionality_reduction import run_dimensionality_reduction
from src.model_training_evaluation import run_model_training_evaluation

def main():
    # Load the dataset
    df = load_dataset()

    # Preprocess and explore the data using Streamlit visualizations
    run_data_exploration()

    # Assume 'DON_concentration' is the target and drop 'hsi_id' if present
    if 'vomitoxin_ppb' in df.columns:
        X = df.drop(['vomitoxin_ppb', 'hsi_id'], axis=1, errors='ignore')
        y = df['vomitoxin_ppb']
    else:
        # If the target column is assumed to be the last column
        X = df.iloc[:, :-1]  # Assume all except the last column are features
        y = df.iloc[:, -1]   # Assume the last column is the target
        print("Assuming target variable is the last column.")

    # Scale features using RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction
    X_reduced, y_reduced = run_dimensionality_reduction(X_scaled, y,desired_components=10)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_reduced, test_size=0.2, random_state=42
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Train and evaluate models
    results = run_model_training_evaluation(X_train, y_train, X_test, y_test)

    # Print model performance comparison
    print("\nModel Performance Comparison:")
    for result in results:
        print(f"{result['model_name']}: MAE = {result['mae']:.4f}, RMSE = {result['rmse']:.4f}, R² = {result['r2']:.4f}")

if __name__ == "__main__":
    main()


