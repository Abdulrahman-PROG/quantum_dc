"""
Energy Consumption Prediction
Predicts data center energy consumption based on workload and environmental factors
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class EnergyPredictor:
    """Energy consumption predictor with polynomial features"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False

    def _create_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create polynomial features from input.

        Features:
            - workload
            - outdoor_temp
            - time_of_day (cyclic encoding)
            - interactions
        """
        workload = X[:, 0]
        outdoor_temp = X[:, 1]
        time_of_day = X[:, 2] if X.shape[1] > 2 else np.zeros(len(X))

        # Create polynomial and interaction features
        features = np.column_stack([
            workload,
            outdoor_temp,
            np.sin(2 * np.pi * time_of_day),  # Cyclic time
            np.cos(2 * np.pi * time_of_day),
            workload ** 2,  # Quadratic workload
            workload * outdoor_temp,  # Interaction
            np.ones(len(X))  # Bias
        ])

        return features

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Train the energy predictor.

        Args:
            X_train: Training features [workload, outdoor_temp, time_of_day, ...]
            y_train: Training labels (energy consumption in kW)
            **kwargs: Additional parameters

        Returns:
            Training metrics
        """
        # Create features
        X_features = self._create_features(X_train)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        # Train model
        alpha = kwargs.get('alpha', 1.0)
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        mse = np.mean((y_pred - y_train) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_train))
        r2 = 1 - (np.sum((y_train - y_pred) ** 2) /
                  np.sum((y_train - np.mean(y_train)) ** 2))

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coefficients': self.model.coef_
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict energy consumption.

        Args:
            X_test: Test features

        Returns:
            Predicted energy consumption (kW)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_features = self._create_features(X_test)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model on test data"""
        y_pred = self.predict(X_test)

        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': y_pred,
            'actuals': y_test
        }
