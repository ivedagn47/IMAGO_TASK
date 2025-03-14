import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

# Importing the necessary functions from your modules
from src.preprocessing import load_dataset
from src.dimensionality_reduction import run_dimensionality_reduction

def load_best_model_from_folder(folder="final"):
    """Load the best model found in the specified folder."""
    if not os.path.exists(folder):
        st.error(f"Folder '{folder}' does not exist.")
        return None, None
        
    for file_name in os.listdir(folder):
        if file_name.endswith(".pkl"):
            model_path = os.path.join(folder, file_name)
            model = joblib.load(model_path)
            st.success(f"Loaded model from {model_path}")
            return model, 'sklearn'
        elif file_name.endswith(".keras"):
            model_path = os.path.join(folder, file_name)
            model = tf.keras.models.load_model(model_path)
            st.success(f"Loaded model from {model_path}")
            return model, 'keras'
    st.error("No saved model found in the specified folder.")
    return None, None

def run_data_exploration_for_prediction(df):
    """Simplified version of run_data_exploration that just returns scaled data without generating plots, suitable for prediction."""
    if 'hsi_id' in df.columns:
        X = df.drop(['hsi_id'], axis=1)
        sample_ids = df['hsi_id']
    else:
        X = df
        sample_ids = np.arange(len(X))
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess_input_data(df):
    """This function applies existing preprocessing and dimensionality reduction."""
    X_scaled = run_data_exploration_for_prediction(df)
    X_reduced, _ = run_dimensionality_reduction(X_scaled, y=None, desired_components=10)
    return X_reduced

def plot_predictions(predictions):
    """Plot predictions in a simple line plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions, label='Predicted Values')
    ax.set_title('Predicted Target Values')
    ax.set_ylabel('Predicted Value')
    ax.set_xlabel('Sample Index')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)  # Passing the figure explicitly

def main():
    st.set_page_config(page_title="Hyperspectral Data Prediction", layout='wide', page_icon="🌟")
    st.markdown("""<style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .streamlit-expanderHeader {
            font-size: 16px !important;
            font-weight: bold;
        }
        </style>""", unsafe_allow_html=True)
    
    st.title("Predict Hyperspectral Target Data")
    st.write("Upload a CSV file with hyperspectral data (without target values) to get predictions.")

    folder = "final"
    model, model_type = load_best_model_from_folder(folder)
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload your spectral data CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        if st.button("Process and Predict", key="predict"):
            with st.spinner("Processing data and making predictions..."):
                X_reduced = preprocess_input_data(df)
                if model_type == 'sklearn':
                    predictions = model.predict(X_reduced)
                elif model_type == 'keras':
                    predictions = model.predict(X_reduced).flatten()
                else:
                    st.error("Unknown model type.")
                    return

                st.subheader("Predictions")
                if 'hsi_id' in df.columns:
                    pred_df = pd.DataFrame({
                        'hsi_id': df['hsi_id'],
                        'DON_concentration_predicted': predictions
                    })
                else:
                    pred_df = pd.DataFrame({
                        'Sample_Index': range(len(predictions)),
                        'DON_concentration_predicted': predictions
                    })
                st.dataframe(pred_df)
                csv = pred_df.to_csv(index=False)
                st.download_button("Download predictions as CSV", csv, "predictions.csv", mime="text/csv")
                plot_predictions(predictions)

if __name__ == "__main__":
    main()
