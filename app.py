import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

def plot_predictions(predictions):
    """Plot predictions in a simple line plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions, marker='o', linestyle='-', label='Predicted Values')
    ax.set_title('Predicted DON Concentration')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Predicted Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def predict_using_api(uploaded_file, api_url):
    """Send data to API and get predictions."""
    try:
        files = {'file': ('data.csv', uploaded_file.getvalue(), 'text/csv')}
        response = requests.post(f"{api_url}/predict/csv", files=files)
        if response.status_code == 200:
            result = response.json()
            return result["predictions"], result.get("sample_ids")
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None, None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="Hyperspectral Data Prediction", layout='wide', page_icon="🌟")
    st.markdown("""<style>
        .big-font { font-size:20px !important; font-weight: bold; }
        .streamlit-expanderHeader { font-size: 16px !important; font-weight: bold; }
        </style>""", unsafe_allow_html=True)
    
    st.title("Predict Hyperspectral DON Concentration")
    st.write("Upload a CSV file with hyperspectral data to get DON concentration predictions using our API.")

    # Fixed API URL for prediction
    api_url = "https://mycotoxin-predictor-api.onrender.com"

    uploaded_file = st.file_uploader("Upload your spectral data CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        if st.button("Process and Predict", key="predict"):
            with st.spinner("Sending data to API and waiting for predictions..."):
                # Reset file pointer before sending to API
                uploaded_file.seek(0)
                predictions, sample_ids = predict_using_api(uploaded_file, api_url)
                if predictions is None:
                    return

            st.subheader("Predictions")
            # Use 'hsi_id' if available, otherwise create default sample IDs
            if 'hsi_id' in df.columns:
                sample_ids = df['hsi_id']
            else:
                sample_ids = list(range(len(predictions)))
            pred_df = pd.DataFrame({
                'Sample_ID': sample_ids,
                'DON_concentration_predicted': predictions
            })
            st.dataframe(pred_df)
            csv = pred_df.to_csv(index=False)
            st.download_button("Download predictions as CSV", csv, "predictions.csv", mime="text/csv")
            plot_predictions(predictions)

if __name__ == "__main__":
    main()
