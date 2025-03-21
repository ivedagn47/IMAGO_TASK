# Hyperspectral Data Analysis for DON Concentration Prediction

## Project Overview
This repository contains the code and resources for predicting Deoxynivalenol (DON) concentration in corn samples using hyperspectral data. Each sample in the dataset is represented by spectral reflectance values across multiple wavelength bands. The project involves data preprocessing, visualization, model training, and deployment of a prediction API.

## Objective
The goal of this project is to:
- Preprocess hyperspectral data to handle missing values, normalize features, and explore potential anomalies.
- Visualize spectral bands to understand data characteristics.
- Train regression models to predict the DON concentration in corn samples.
- Develop and deploy a production-ready prediction API with a front-end interface for real-time predictions.

## Data Description
The dataset features hyperspectral reflectance data from corn samples, with each row representing a sample. Columns include:
- **Reflectance Features**: Continuous variables representing reflectance at various wavelengths.
- **DON Concentration**: Target variable, continuous, representing DON concentration in ppm.

## Installation
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/ivedagn47/IMAGO_TASK.git
pip install -r requirements.txt
pip install -r requirements-api.txt
```

## Repository Structure
```
.
├── data/                  # Dataset directory
├── final/                 # Final trained models and scalers
├── nn_tuning/             # Neural network tuning files
├── plots/                 # Generated plots for analysis and evaluation
├── src/                   # Source code for the project
│   ├── dimensionality_reduction.py
│   ├── interpretability.py
│   ├── model_training_evaluation.py
│   ├── preprocessing.py
│   └── utils/
│       └── logger.py
├── requirements.txt       # Python dependencies for the project
├── requirements-api.txt   # Additional dependencies for the API
├── train.py               # Script for training models
├── api.py                 # FastAPI server script
└── app.py                 # Streamlit application script
```

## Usage

### Model Preprocessing and Training 
Train the model using:
```bash
python train.py
```
This script trains multiple models and saves the best performing model to the `final/` directory.

### Launching the API
Start the FastAPI server:
```bash
uvicorn api:app --reload
```
After running, the server will be accessible at `http://127.0.0.1:8000`. Copy this URL and update it in the Streamlit app (`app.py`) if needed.

### Running the Streamlit App
Edit the `app.py` file to update the API URL variable with your local server's URL. Then, run the Streamlit application:
```bash
streamlit run app.py
```
You can also access the deployed Streamlit app on Render here: [Streamlit App](https://imagotask-xmlgzv3pkcapdlyzhhtyyb.streamlit.app/)

![image](https://github.com/user-attachments/assets/e94bae68-3872-403e-8854-9cea0db476b3)


## Deployment on Render
This project has been deployed using Render. You can use Render to host your own version of this project if desired.

## Documentation
Detailed docstrings are provided for each function and module, explaining their purpose and parameters.

## Contributing
Contributions are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License.
