# src/dimensionality_reduction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Import the logger from utils
from src.utils.logger import setup_logger
logger = setup_logger('dimensionality_reduction')

def run_dimensionality_reduction(X_scaled, y, desired_components=10):
    """
    Performs PCA and t-SNE on the scaled data, visualizes the explained variance,
    scatter plots, and returns the dimensionally reduced data with a fixed number of components.
    
    Parameters:
        X_scaled (ndarray or DataFrame): Scaled feature data.
        y (ndarray or Series): Target variable.
        desired_components (int): Number of PCA components to retain (default is 10).
        
    Returns:
        X_model (ndarray): PCA-reduced data using the first `desired_components` components.
        y: The original target variable.
    """
    
    logger.info("Starting dimensionality reduction")
    
    # Step 1: Perform PCA to reduce dimensionality
    n_components = min(20, X_scaled.shape[1])  # Ensure that the number of components does not exceed the number of features
    logger.info(f"Performing PCA with n_components = {n_components}")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)  # Apply PCA to the scaled data
    logger.info("PCA transformation complete")
    
    # Step 2: Calculate explained variance and cumulative variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)  # Cumulative explained variance across components
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # Find number of components for 95% variance
    logger.info(f"Cumulative variance computed; {n_components_95} components cover 95% variance")
    
    # Step 3: Plot cumulative explained variance vs. number of components
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.axvline(x=n_components_95, color='g', linestyle='--',
                label=f'{n_components_95} Components for 95% Variance')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.savefig('pca_variance.png')  # Save the plot to a file
    logger.info("Saved PCA variance plot as 'pca_variance.png'")
    
    # Step 4: Scatter plot of the first two principal components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Target Variable')  # Color bar to indicate target variable values
    plt.title('PCA: First Two Principal Components')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('pca_2d.png')  # Save the scatter plot to a file
    logger.info("Saved 2D PCA scatter plot as 'pca_2d.png'")
    
    # Step 5: Optional 3D visualization if at least three components are available
    if n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # Importing here to avoid issues if not needed
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                             c=y, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, label='Target Variable')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('3D PCA Visualization')
        plt.savefig('pca_3d.png')  # Save the 3D plot to a file
        logger.info("Saved 3D PCA plot as 'pca_3d.png'")
    
    # Step 6: Visualize the loading weights for the top 3 principal components
    plt.figure(figsize=(12, 8))
    for i in range(min(3, len(pca.components_))):  # Plot the first 3 components or fewer if less are available
        plt.plot(pca.components_[i], label=f'PC{i+1}')
    plt.title('Loading Weights of Top 3 Principal Components')
    plt.xlabel('Feature Index')
    plt.ylabel('Loading Weight')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('pca_loadings.png')  # Save the loading weights plot to a file
    logger.info("Saved PCA loadings plot as 'pca_loadings.png'")
    
    # Step 7: Perform t-SNE visualization using the PCA components covering 95% variance
    optimal_perplexity = min(30, len(X_scaled) // 5)  # Set perplexity based on the dataset size
    logger.info(f"Performing t-SNE with optimal perplexity = {optimal_perplexity} on {n_components_95} PCA components")
    tsne = TSNE(n_components=2, random_state=42, perplexity=optimal_perplexity)
    X_tsne = tsne.fit_transform(X_pca[:, :n_components_95])  # Apply t-SNE on the selected PCA components
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Target Variable')
    plt.title(f't-SNE Visualization (Perplexity = {optimal_perplexity})')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('tsne_2d.png')  # Save the t-SNE plot to a file
    logger.info("Saved t-SNE plot as 'tsne_2d.png'")
    
    # Step 8: Ensure desired_components does not exceed the number of available PCA components
    if desired_components > X_pca.shape[1]:
        logger.warning(f"desired_components ({desired_components}) is greater than available components ({X_pca.shape[1]}). Using {X_pca.shape[1]} components instead.")
        desired_components = X_pca.shape[1]  # Adjust to available components if necessary
    
    # Step 9: Return the first 'desired_components' principal components
    X_model = X_pca[:, :desired_components]
    logger.info(f"Dimensionality reduction complete. Returning data with shape {X_model.shape}")
    return X_model, y  # Return the reduced dataset along with the original target variable
