import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime

class SignalModel:
    def __init__(self):
        self.model = None

    def create_model(self, input_shape):
        """Create a more robust and complex model for signal classification"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("Signal classification model created.")

    def train_model(self, X_train, y_train, epochs=20, batch_size=32, callbacks=None):
        """Train the model with provided data"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        logging.info("Model trained successfully.")

    def predict_signal(self, signal_data):
        """Predict if the signal is present"""
        if self.model is None:
            raise ValueError("Model not created.")
        
        signal_data = np.array(signal_data).reshape(1, -1)  # Reshape for prediction
        prediction = self.model.predict(signal_data)[0][0]
        return "Signal" if prediction >= 0.5 else "Noise"

    def save_model(self, path):
        """Save the trained model to disk"""
        if not path.endswith('.h5'):
            path += '.h5'
        self.model.save(path)
        logging.info(f"Model saved to {path}.")

    def load_model(self, path):
        """Load a model from disk"""
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            logging.info(f"Model loaded from {path}.")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")

    def save_training_data(self, X_train, y_train, path):
        """Save training data for future use"""
        np.savez(path, X_train=X_train, y_train=y_train)
        logging.info(f"Training data saved to {path}.")

    def load_training_data(self, path):
        """Load training data from disk"""
        if os.path.exists(path):
            data = np.load(path)
            X_train = data['X_train']
            y_train = data['y_train']
            logging.info(f"Training data loaded from {path}.")
            return X_train, y_train
        else:
            raise FileNotFoundError(f"Training data file not found at {path}")
