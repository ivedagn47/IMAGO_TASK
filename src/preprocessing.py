# src/preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from src.utils.logger import setup_logger

logger = setup_logger('preprocessing')

def load_dataset():
    """
    Loads the dataset from a CSV file.
    
    Returns:
        df (DataFrame): Loaded dataset.
    """
    logger.info("Loading dataset from 'data/dataset.csv'")
    df = pd.read_csv('data/dataset.csv')
    logger.info(f"Dataset loaded with shape {df.shape}")
    return df

def run_data_exploration():
    """
    Perform data exploration on the dataset.
    This includes data preview, separation of features and target variable,
    and various visualizations such as spectral reflectance curves, heatmaps, and scaling.
    """
    logger.info("Starting data exploration")
    # Load data
    df = load_dataset()
    logger.info("### Dataset Preview")
    logger.info(f"\n{df.head()}")
    print("### Dataset Preview")
    print(df.head())

    # Separate features and target variable
    if 'DON_concentration' in df.columns:
        logger.info("Found 'DON_concentration' in dataset columns. Separating target variable.")
        X = df.drop(['DON_concentration', 'hsi_id'], axis=1, errors='ignore')
        y = df['DON_concentration']
    else:
        logger.warning("Target variable 'DON_concentration' not found explicitly. Assuming last column as target.")
        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        print("Assuming target variable is the last column and hsi_id is the first column.")

    # Extract sample IDs if available
    if 'hsi_id' in df.columns:
        sample_ids = df['hsi_id']
        logger.info("Sample IDs extracted from 'hsi_id' column.")
    else:
        sample_ids = np.arange(len(X))
        logger.warning("No explicit sample IDs found. Using index as sample ID.")
        print("No explicit sample IDs found. Using index as sample ID.")

    # === DATA VISUALIZATION ===

    # Spectral Reflectance Curves for First 5 Samples
    print("#### Spectral Reflectance Curves for First 5 Samples")
    logger.info("Plotting Spectral Reflectance Curves for First 5 Samples")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for i in range(min(5, len(X))):
        ax1.plot(X.iloc[i], label=f'Sample {i}')
    ax1.set_title('Spectral Reflectance Curves for First 5 Samples')
    ax1.set_xlabel('Wavelength Band Index')
    ax1.set_ylabel('Reflectance')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    logger.info("Displayed Spectral Reflectance Curves for First 5 Samples")

    # Heatmap of Spectral Data (First 20 Samples)
    print("#### Heatmap of Spectral Data (First 20 Samples)")
    logger.info("Plotting Heatmap of Spectral Data for First 20 Samples")
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    sns.heatmap(X.iloc[:20], cmap='viridis', ax=ax2)
    ax2.set_title('Heatmap of Spectral Data (First 20 Samples)')
    ax2.set_xlabel('Wavelength Band')
    ax2.set_ylabel('Sample Index')
    plt.show()
    logger.info("Displayed Heatmap of Spectral Data for First 20 Samples")

    # Average Spectral Signature with Standard Deviation
    print("#### Average Spectral Signature with Standard Deviation")
    logger.info("Plotting Average Spectral Signature with Standard Deviation")
    avg_spectrum = X.mean(axis=0)
    std_spectrum = X.std(axis=0)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(avg_spectrum, color='blue', label='Mean Reflectance')
    ax3.fill_between(range(len(avg_spectrum)),
                     avg_spectrum - std_spectrum,
                     avg_spectrum + std_spectrum,
                     alpha=0.2, color='blue')
    ax3.set_title('Average Spectral Signature with Standard Deviation')
    ax3.set_xlabel('Wavelength Band Index')
    ax3.set_ylabel('Reflectance')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    logger.info("Displayed Average Spectral Signature with Standard Deviation plot")

    # Feature scaling (Robust Scaling)
    logger.info("Performing Robust Scaling on features")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Robust Scaling complete")
    print("#### Data after Robust Scaling")
    print("Median:", np.median(X_scaled, axis=0))
    print("IQR:", np.percentile(X_scaled, 75, axis=0) - np.percentile(X_scaled, 25, axis=0))
    logger.info("Displayed median and IQR of scaled data")

    return X_scaled, y, sample_ids
