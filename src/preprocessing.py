# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

def load_dataset():
    """
    Loads the dataset from a CSV file.
    
    Returns:
        df (DataFrame): Loaded dataset.
    """
    # Adjust the file path if needed
    df = pd.read_csv('data/dataset.csv')
    return df

def run_data_exploration():
    """
    Perform data exploration on the dataset.
    This includes data preview, separation of features and target variable,
    and various visualizations such as spectral reflectance curves, heatmaps, and scaling.
    """
    # Load data
    df = load_dataset()
    print("### Dataset Preview")
    print(df.head())

    # Separate features and target variable
    # Checking if the target variable 'DON_concentration' is present in the dataset
    if 'DON_concentration' in df.columns:
        # If the target variable is present, separate it from the features
        X = df.drop(['DON_concentration', 'hsi_id'], axis=1, errors='ignore')  # Drop target and sample ID if available
        y = df['DON_concentration']
    else:
        # If 'DON_concentration' is not explicitly provided, assume target is the last column
        X = df.iloc[:, 1:-1]  # Skip the first column (assumed to be sample ID) and last column (assumed target)
        y = df.iloc[:, -1]
        print("Assuming target variable is the last column and hsi_id is the first column.")

    # Extract sample IDs if available
    if 'hsi_id' in df.columns:
        sample_ids = df['hsi_id']
    else:
        # If no explicit sample IDs are available, use the row index as sample ID
        sample_ids = np.arange(len(X))
        print("No explicit sample IDs found. Using index as sample ID.")

    # === DATA VISUALIZATION ===
    
    # Spectral Reflectance Curves for First 5 Samples
    print("#### Spectral Reflectance Curves for First 5 Samples")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for i in range(min(5, len(X))):
        ax1.plot(X.iloc[i], label=f'Sample {i}')
    ax1.set_title('Spectral Reflectance Curves for First 5 Samples')
    ax1.set_xlabel('Wavelength Band Index')
    ax1.set_ylabel('Reflectance')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.show()  # Show the plot

    # Heatmap of Spectral Data (First 20 Samples)
    print("#### Heatmap of Spectral Data (First 20 Samples)")
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    sns.heatmap(X.iloc[:20], cmap='viridis', ax=ax2)
    ax2.set_title('Heatmap of Spectral Data (First 20 Samples)')
    ax2.set_xlabel('Wavelength Band')
    ax2.set_ylabel('Sample Index')
    plt.show()  # Show the plot

    # Average Spectral Signature with Standard Deviation
    print("#### Average Spectral Signature with Standard Deviation")
    avg_spectrum = X.mean(axis=0)  # Mean of each wavelength band across all samples
    std_spectrum = X.std(axis=0)  # Standard deviation of each wavelength band
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
    plt.show()  # Show the plot

    # Feature scaling (robust scaling)
    # RobustScaler handles outliers by using median and interquartile range for scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)  # Apply robust scaling to features
    print("#### Data after Robust Scaling")
    print("Median:", np.median(X_scaled, axis=0))  # Display median values after scaling
    print("IQR:", np.percentile(X_scaled, 75, axis=0) - np.percentile(X_scaled, 25, axis=0))  # Display IQR values after scaling

