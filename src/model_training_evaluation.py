# src/model_training_evaluation.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, BatchNormalization

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

import joblib
import xgboost as xgb

# Update: Import keras_tuner (formerly kerastuner)
try:
    from keras_tuner import RandomSearch
except ImportError:
    print("Please install keras_tuner: pip install keras-tuner")

# Import logger from utils
from src.utils.logger import setup_logger
logger = setup_logger('model_training_evaluation')

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a regression model and display performance metrics.
    Also saves an actual vs predicted plot.
    """
    logger.info(f"Evaluating model: {model_name}")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        logger.error(f"Error during prediction in {model_name}: {e}")
        return None

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"===== {model_name} Performance =====")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")

    # Create and save actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure the output directory exists
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    plot_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_predictions.png')
    plt.savefig(plot_path)
    logger.info(f"Saved {model_name} prediction plot as '{plot_path}'")
     
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred
    }


def save_model_generic(model, file_path, is_keras_model=False, extra_objects=None):
    """
    Save a model to disk. For non-Keras models, use joblib; for Keras models, use model.save.
    extra_objects: a dictionary of additional objects (e.g., scalers) to save.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory for saving model: {directory}")
    try:
        if is_keras_model:
            model.save(file_path)
            logger.info(f"Keras model saved to {file_path}")
        else:
            joblib.dump(model, file_path)
            logger.info(f"Model saved to {file_path}")
        if extra_objects:
            for obj, path in extra_objects.items():
                obj_dir = os.path.dirname(path)
                if obj_dir and not os.path.exists(obj_dir):
                    os.makedirs(obj_dir)
                    logger.info(f"Created directory for saving extra object: {obj_dir}")
                joblib.dump(obj, path)
                logger.info(f"Extra object saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


def run_model_training_evaluation(X_train, y_train, X_test, y_test):
    """
    Runs the training and evaluation of various models:
    Random Forest, XGBoost, Neural Network, LSTM, Transformer with Attention, and CNN.
    Finally performs model comparison, saving, and interpretation.
    """
    # ----------------- Random Forest -----------------
    logger.info("Starting training for Random Forest model")
    print("\n1. Training Random Forest model...")

    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf_base = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_grid,
        n_iter=20,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    rf_random.fit(X_train, y_train)
    logger.info(f"Random Forest best parameters: {rf_random.best_params_}")
    logger.info(f"Random Forest best cross-validation score: {rf_random.best_score_:.4f}")

    rf_best = rf_random.best_estimator_
    rf_results = evaluate_model(rf_best, X_test, y_test, "Random Forest")

    # Plot feature importances if available
    if hasattr(rf_best, 'feature_importances_'):
        importances = rf_best.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Random Forest Feature Importance (Principal Components)')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [f'PC{i+1}' for i in indices])
        plt.tight_layout()
        rf_feat_path = os.path.join("plots", 'rf_feature_importance.png')
        plt.savefig(rf_feat_path)
        logger.info(f"Saved Random Forest feature importance plot as '{rf_feat_path}'")
         

    # ----------------- XGBoost -----------------
    logger.info("Starting training for XGBoost model")
    print("\n2. Training XGBoost model...")

    xgb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_base = xgb.XGBRegressor(random_state=42)
    xgb_random = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_grid,
        n_iter=20,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    xgb_random.fit(X_train, y_train)
    logger.info(f"XGBoost best parameters: {xgb_random.best_params_}")
    logger.info(f"XGBoost best cross-validation score: {xgb_random.best_score_:.4f}")

    xgb_best = xgb_random.best_estimator_
    xgb_results = evaluate_model(xgb_best, X_test, y_test, "XGBoost")

    # ----------------- Neural Network -----------------
    logger.info("Starting training for Neural Network model")
    print("\n3. Training Neural Network model...")

    # Scale target variable
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    def create_nn_model(input_dim, hidden_layers, neurons, dropout_rate, activation, learning_rate):
        model = Sequential()
        model.add(Dense(neurons, input_dim=input_dim, activation=activation))
        model.add(Dropout(dropout_rate))
        for _ in range(hidden_layers):
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        return model

    # Define model building function for tuner
    def build_model(hp):
        input_dim = X_train.shape[1]
        hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=3, step=1)
        neurons = hp.Int('neurons', min_value=16, max_value=128, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
        learning_rate = hp.Float('learning_rate', min_value=0.001, max_value=0.01, step=0.001)
        return create_nn_model(input_dim, hidden_layers, neurons, dropout_rate, activation, learning_rate)

    try:
        tuner = RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=2,
            directory='nn_tuning',
            project_name='hyperspectral_regression'
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        tuner.search(
            X_train, y_train_scaled,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        best_hp = tuner.get_best_hyperparameters()[0]
        logger.info("Best Neural Network parameters:")
        logger.info(f"Hidden layers: {best_hp.get('hidden_layers')}")
        logger.info(f"Neurons: {best_hp.get('neurons')}")
        logger.info(f"Dropout rate: {best_hp.get('dropout_rate')}")
        logger.info(f"Activation: {best_hp.get('activation')}")
        logger.info(f"Learning rate: {best_hp.get('learning_rate')}")
        nn_model = build_model(best_hp)
    except Exception as e:
        logger.error("KerasTuner not available or an error occurred. Using default Neural Network parameters.", exc_info=True)
        input_dim = X_train.shape[1]
        nn_model = create_nn_model(input_dim, hidden_layers=2, neurons=64, dropout_rate=0.2, activation='relu', learning_rate=0.001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    history = nn_model.fit(
        X_train, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    logger.info("Neural Network training complete. Saving training history plot.")
    # Plot training history for NN
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()

    if not os.path.exists("plots"):
        os.makedirs("plots")
        logger.info("Created 'plots' directory.")
    nn_history_path = os.path.join("plots", 'nn_training_history.png')
    plt.savefig(nn_history_path)
    logger.info(f"Saved Neural Network training history plot as '{nn_history_path}'")
     

    y_pred_scaled = nn_model.predict(X_test).flatten()
    y_pred_nn = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    r2_nn = r2_score(y_test, y_pred_nn)

    logger.info("===== Neural Network Performance =====")
    logger.info(f"Mean Absolute Error: {mae_nn:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse_nn:.4f}")
    logger.info(f"R² Score: {r2_nn:.4f}")

    # Plot NN predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_nn, alpha=0.7)
    min_val = min(y_test.min(), min(y_pred_nn))
    max_val = max(y_test.max(), max(y_pred_nn))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Neural Network: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    nn_pred_path = os.path.join("plots", 'neural_network_predictions.png')
    plt.savefig(nn_pred_path)
    logger.info(f"Saved Neural Network predictions plot as '{nn_pred_path}'")
     

    nn_results = {
        'model_name': 'Neural Network',
        'mae': mae_nn,
        'rmse': rmse_nn,
        'r2': r2_nn,
        'y_pred': y_pred_nn
    }

    # ----------------- LSTM Model -----------------
    logger.info("Starting training for LSTM model")
    print("\n4. Training LSTM model...")

    # Reshape for LSTM input: (samples, time_steps, features)
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    def create_lstm_model(input_shape, lstm_units=64, dense_units=32, dropout_rate=0.2):
        model = Sequential()
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))  # Using Bidirectional LSTM
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(lstm_units // 2)))  # Using Bidirectional LSTM
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    lstm_model = create_lstm_model(input_shape=(X_train_lstm.shape[1], 1))
    lstm_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    lstm_history = lstm_model.fit(
        X_train_lstm, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    logger.info("LSTM training complete. Saving training history plot.")
    # Plot LSTM training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['loss'], label='Train Loss')
    plt.plot(lstm_history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history.history['mae'], label='Train MAE')
    plt.plot(lstm_history.history['val_mae'], label='Val MAE')
    plt.title('LSTM Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    lstm_history_path = os.path.join("plots", 'lstm_training_history.png')
    plt.savefig(lstm_history_path)
    logger.info(f"Saved LSTM training history plot as '{lstm_history_path}'")
     

    y_pred_lstm_scaled = lstm_model.predict(X_test_lstm).flatten()
    y_pred_lstm = y_scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()

    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    r2_lstm = r2_score(y_test, y_pred_lstm)

    logger.info("===== LSTM Performance =====")
    logger.info(f"Mean Absolute Error: {mae_lstm:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse_lstm:.4f}")
    logger.info(f"R² Score: {r2_lstm:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lstm, alpha=0.7)
    min_val = min(y_test.min(), min(y_pred_lstm))
    max_val = max(y_test.max(), max(y_pred_lstm))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('LSTM: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    lstm_pred_path = os.path.join("plots", 'lstm_predictions.png')
    plt.savefig(lstm_pred_path)
    logger.info(f"Saved LSTM predictions plot as '{lstm_pred_path}'")
     

    lstm_results = {
        'model_name': 'LSTM',
        'mae': mae_lstm,
        'rmse': rmse_lstm,
        'r2': r2_lstm,
        'y_pred': y_pred_lstm
    }

    # ----------------- Transformer with Attention Model -----------------
    logger.info("Starting training for Transformer with Attention model")
    print("\n5. Training Transformer with Attention model...")

    # Reshape data for transformer (3D input)
    X_train_attn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_attn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Custom attention layer
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="attention_weight",
                                     shape=(input_shape[-1], 1),
                                     initializer="normal")
            self.b = self.add_weight(name="attention_bias",
                                     shape=(input_shape[1], 1),
                                     initializer="zeros")
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
            e = tf.keras.backend.squeeze(e, axis=-1)
            alpha = tf.keras.backend.softmax(e)
            alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
            context = x * alpha
            context = tf.keras.backend.sum(context, axis=1)
            return context

    def create_transformer_model(input_shape, head_size=64, num_heads=2, ff_dim=64, dropout_rate=0.2):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(2):  # Two transformer blocks
            attention_output = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
            )(x, x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            ffn = tf.keras.Sequential([
                Dense(ff_dim, activation="relu"),
                Dense(input_shape[-1])
            ])
            x = ffn(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(dropout_rate)(x)
        x = AttentionLayer()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    transformer_model = create_transformer_model(input_shape=(X_train_attn.shape[1], 1))
    transformer_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    transformer_history = transformer_model.fit(
        X_train_attn, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    logger.info("Transformer with Attention training complete. Saving training history plot.")
    # Plot transformer training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(transformer_history.history['loss'], label='Train Loss')
    plt.plot(transformer_history.history['val_loss'], label='Val Loss')
    plt.title('Transformer Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(transformer_history.history['mae'], label='Train MAE')
    plt.plot(transformer_history.history['val_mae'], label='Val MAE')
    plt.title('Transformer Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    transformer_history_path = os.path.join("plots", 'transformer_training_history.png')
    plt.savefig(transformer_history_path)
    logger.info(f"Saved Transformer training history plot as '{transformer_history_path}'")
     

    y_pred_transformer_scaled = transformer_model.predict(X_test_attn).flatten()
    y_pred_transformer = y_scaler.inverse_transform(y_pred_transformer_scaled.reshape(-1, 1)).flatten()

    mae_transformer = mean_absolute_error(y_test, y_pred_transformer)
    rmse_transformer = np.sqrt(mean_squared_error(y_test, y_pred_transformer))
    r2_transformer = r2_score(y_test, y_pred_transformer)

    logger.info("===== Transformer with Attention Performance =====")
    logger.info(f"Mean Absolute Error: {mae_transformer:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse_transformer:.4f}")
    logger.info(f"R² Score: {r2_transformer:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_transformer, alpha=0.7)
    min_val = min(y_test.min(), min(y_pred_transformer))
    max_val = max(y_test.max(), max(y_pred_transformer))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Transformer with Attention: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    transformer_pred_path = os.path.join("plots", 'transformer_predictions.png')
    plt.savefig(transformer_pred_path)
    logger.info(f"Saved Transformer predictions plot as '{transformer_pred_path}'")
     

    transformer_results = {
        'model_name': 'Transformer with Attention',
        'mae': mae_transformer,
        'rmse': rmse_transformer,
        'r2': r2_transformer,
        'y_pred': y_pred_transformer
    }

    # ----------------- CNN Model -----------------
    logger.info("Starting training for CNN model")
    print("\nTraining CNN model for regression...")

    # Assume X_train from PCA has shape (samples, n_components).
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    def create_cnn_model(input_shape, num_filters=64, kernel_size=3, dropout_rate=0.2, dense_units=64):
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', 
                        padding='same', input_shape=input_shape))
        model.add(BatchNormalization())  # Adding Batch Normalization
        model.add(Dropout(dropout_rate))
        model.add(Conv1D(filters=num_filters*2, kernel_size=kernel_size, activation='relu', 
                        padding='same'))
        model.add(BatchNormalization())  # Adding Batch Normalization
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    cnn_model = create_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
    cnn_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

    cnn_history = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    logger.info("CNN training complete. Evaluating CNN model.")
    cnn_results = evaluate_model(cnn_model, X_test_cnn, y_test, "CNN")

    # Plot training history for CNN
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['loss'], label='Train Loss')
    plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['mae'], label='Train MAE')
    plt.plot(cnn_history.history['val_mae'], label='Validation MAE')
    plt.title('CNN Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    cnn_history_path = os.path.join("plots", 'cnn_training_history.png')
    plt.savefig(cnn_history_path)
    logger.info(f"Saved CNN training history plot as '{cnn_history_path}'")
     

    # Make predictions with the CNN model
    y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()

    mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
    rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
    r2_cnn = r2_score(y_test, y_pred_cnn)

    logger.info("===== CNN Performance =====")
    logger.info(f"Mean Absolute Error: {mae_cnn:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse_cnn:.4f}")
    logger.info(f"R² Score: {r2_cnn:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_cnn, alpha=0.7)
    min_val = min(y_test.min(), y_pred_cnn.min())
    max_val = max(y_test.max(), y_pred_cnn.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('CNN: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    cnn_pred_path = os.path.join("plots", 'cnn_predictions.png')
    plt.savefig(cnn_pred_path)
    logger.info(f"Saved CNN predictions plot as '{cnn_pred_path}'")
     

    # Build tuned CNN model using keras_tuner
    def build_cnn_model_tuned(hp):
        inputs = tf.keras.Input(shape=(X_train_cnn.shape[1], 1))
        num_filters = hp.Choice('num_filters', values=[32, 64, 128])
        kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        dense_units = hp.Choice('dense_units', values=[32, 64, 128])
        
        x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
        x = Dropout(dropout_rate)(x)
        x = Conv1D(filters=num_filters * 2, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    import keras_tuner as kt

    tuner = kt.RandomSearch(
        build_cnn_model_tuned,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='cnn_tuning',
        project_name='cnn_hyperspectral',
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train_cnn, y_train,
                 epochs=50,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[early_stopping],
                 verbose=1)

    best_cnn_model = tuner.get_best_models(num_models=1)[0]
    history = best_cnn_model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    tuned_cnn_results = evaluate_model(best_cnn_model, X_test_cnn, y_test, "Tuned CNN")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('CNN Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    tuned_cnn_history_path = os.path.join("plots", 'cnn_tuned_training_history.png')
    plt.savefig(tuned_cnn_history_path)
    logger.info(f"Saved Tuned CNN training history plot as '{tuned_cnn_history_path}'")
     

    y_pred_cnn = best_cnn_model.predict(X_test_cnn).flatten()

    mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
    rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
    r2_cnn = r2_score(y_test, y_pred_cnn)

    logger.info("===== Tuned CNN Performance =====")
    logger.info(f"Mean Absolute Error: {mae_cnn:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse_cnn:.4f}")
    logger.info(f"R² Score: {r2_cnn:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_cnn, alpha=0.7)
    min_val = min(y_test.min(), y_pred_cnn.min())
    max_val = max(y_test.max(), y_pred_cnn.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Tuned CNN: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    tuned_cnn_pred_path = os.path.join("plots", 'cnn_tuned_predictions.png')
    plt.savefig(tuned_cnn_pred_path)
    logger.info(f"Saved Tuned CNN predictions plot as '{tuned_cnn_pred_path}'")
     

    # ===== MODEL COMPARISON =====
    logger.info("Starting model comparison")
    print("\n===== MODEL COMPARISON =====")

    all_results = [rf_results, xgb_results, nn_results, lstm_results, transformer_results, cnn_results, tuned_cnn_results]

    comparison_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'MAE': result['mae'],
            'RMSE': result['rmse'],
            'R²': result['r2']
        }
        for result in all_results
    ])

    logger.info("Model Performance Comparison:")
    logger.info(f"\n{comparison_df}")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.bar(comparison_df['Model'], comparison_df['MAE'])
    plt.title('Mean Absolute Error (Lower is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.bar(comparison_df['Model'], comparison_df['RMSE'])
    plt.title('Root Mean Squared Error (Lower is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.bar(comparison_df['Model'], comparison_df['R²'])
    plt.title('R² Score (Higher is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.scatter(y_test, rf_results['y_pred'], alpha=0.5, label='Random Forest')
    plt.scatter(y_test, xgb_results['y_pred'], alpha=0.5, label='XGBoost')
    plt.scatter(y_test, nn_results['y_pred'], alpha=0.5, label='Neural Network')
    plt.scatter(y_test, lstm_results['y_pred'], alpha=0.5, label='LSTM')
    plt.scatter(y_test, transformer_results['y_pred'], alpha=0.5, label='transformer')
    min_val = min(y_test.min(), min([result['y_pred'].min() for result in all_results]))
    max_val = max(y_test.max(), max([result['y_pred'].max() for result in all_results]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.title('All Models: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    comparison_plot_path = 'model_comparison.png'
    plt.savefig(comparison_plot_path)
    logger.info(f"Saved model comparison plot as '{comparison_plot_path}'")
     

    best_model_idx = comparison_df['R²'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    logger.info(f"Best performing model based on R² score: {best_model}")
    print(f"\nBest performing model based on R² score: {best_model}")

    # ===== SAVE BEST MODEL =====
    if not os.path.exists('final'):
        os.makedirs('final')
        logger.info("Created 'final' directory for saving best model.")

    if best_model == 'Random Forest':
        joblib.dump(rf_best, os.path.join('final', 'best_model_rf.pkl'))
        logger.info("Saved Random Forest model to 'final/best_model_rf.pkl'")
    elif best_model == 'XGBoost':
        joblib.dump(xgb_best, os.path.join('final', 'best_model_xgb.pkl'))
        logger.info("Saved XGBoost model to 'final/best_model_xgb.pkl'")
    elif best_model == 'Neural Network':
        nn_model.save(os.path.join('final', 'best_model_nn.keras'))
        joblib.dump(y_scaler, os.path.join('final', 'y_scaler.pkl'))
        logger.info("Saved Neural Network model to 'final/best_model_nn.keras'")
        logger.info("Saved target scaler to 'final/y_scaler.pkl'")
    elif best_model == 'LSTM':
        lstm_model.save(os.path.join('final', 'best_model_lstm.keras'))
        joblib.dump(y_scaler, os.path.join('final', 'y_scaler.pkl'))
        logger.info("Saved LSTM model to 'final/best_model_lstm.keras'")
        logger.info("Saved target scaler to 'final/y_scaler.pkl'")
    elif best_model == 'Transformer with Attention':
        transformer_model.save(os.path.join('final', 'best_model_transformer.keras'))
        joblib.dump(y_scaler, os.path.join('final', 'y_scaler.pkl'))
        logger.info("Saved Transformer model to 'final/best_model_transformer.keras'")
        logger.info("Saved target scaler to 'final/y_scaler.pkl'")
    elif best_model == 'CNN':
        cnn_model.save(os.path.join('final', 'best_model_cnn.keras'))
        logger.info("Saved CNN model to 'final/best_model_cnn.keras'")
    elif best_model == 'Tuned CNN':
        best_cnn_model.save(os.path.join('final', 'best_model_tuned_cnn.keras'))
        logger.info("Saved Tuned CNN model to 'final/best_model_tuned_cnn.keras'")
        
    # ===== MODEL INTERPRETATION (FOR TREE-BASED MODELS) =====
    if best_model in ['Random Forest', 'XGBoost']:
        logger.info("Starting feature importance analysis for tree-based model")
        print("\n===== FEATURE IMPORTANCE ANALYSIS =====")

        if best_model == 'Random Forest':
            importance = rf_best.feature_importances_
            model_obj = rf_best
        else:  # XGBoost
            importance = xgb_best.feature_importances_
            model_obj = xgb_best

        feature_importance = pd.DataFrame({
            'Feature': [f'PC{i+1}' for i in range(len(importance))],
            'Importance': importance
        })

        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        logger.info("Top 5 important features:")
        logger.info(f"\n{feature_importance.head(5)}")

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title(f'{best_model} Feature Importance')
        plt.tight_layout()
        fi_path = f'{best_model.lower().replace(" ", "_")}_importance.png'
        plt.savefig(fi_path)
        logger.info(f"Saved feature importance plot as '{fi_path}'")
    from src.interpretability import interpret_model_with_shap

    # 2. Figure out which model object is best
    if best_model == 'Random Forest':
        best_model_obj = rf_best
    elif best_model == 'XGBoost':
        best_model_obj = xgb_best
    elif best_model == 'Neural Network':
        best_model_obj = nn_model
    elif best_model == 'LSTM':
        best_model_obj = lstm_model
    elif best_model == 'Transformer with Attention':
        best_model_obj = transformer_model
    elif best_model == 'CNN':
        best_model_obj = cnn_model
    elif best_model == 'Tuned CNN':
        best_model_obj = best_cnn_model

    # 3. Call SHAP interpretability
    interpret_model_with_shap(
        best_model_name=best_model,  # e.g. "Random Forest"
        best_model=best_model_obj,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_scaler=y_scaler  # pass if you used a MinMaxScaler for the target in your NN
    )    
    return {
        'Random Forest': rf_results,
        'XGBoost': xgb_results,
        'Neural Network': nn_results,
        'LSTM': lstm_results,
        'Transformer with Attention': transformer_results
    }
