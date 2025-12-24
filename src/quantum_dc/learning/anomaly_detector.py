"""
Anomaly Detection for Data Center Operations
Detects abnormal conditions using quantum-inspired SVM
"""

import numpy as np
from typing import Dict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """SVM-based anomaly detector"""

    def __init__(self, kernel: str = 'rbf'):
        """
        Initialize anomaly detector.

        Args:
            kernel: SVM kernel ('rbf', 'linear', 'poly')
        """
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Train anomaly detector.

        Args:
            X_train: Training features [temp, power, voltage, ...]
            y_train: Labels (0=normal, 1=anomaly)
            **kwargs: SVM parameters

        Returns:
            Training metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train SVM
        self.model = SVC(
            kernel=self.kernel,
            gamma=kwargs.get('gamma', 'auto'),
            C=kwargs.get('C', 1.0),
            probability=True
        )
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        # Training metrics
        y_pred = self.model.predict(X_scaled)
        accuracy = np.mean(y_pred == y_train)

        # Confusion matrix
        tp = np.sum((y_pred == 1) & (y_train == 1))
        fp = np.sum((y_pred == 1) & (y_train == 0))
        tn = np.sum((y_pred == 0) & (y_train == 0))
        fn = np.sum((y_pred == 0) & (y_train == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }

    def predict(
        self,
        X_test: np.ndarray,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Detect anomalies in new data.

        Args:
            X_test: Test features
            return_proba: If True, return probabilities

        Returns:
            Predictions (0=normal, 1=anomaly) or probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        if return_proba:
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def detect(
        self,
        X_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Detect anomalies with detailed output.

        Args:
            X_test: Test data
            threshold: Probability threshold for anomaly

        Returns:
            Detection results with confidence scores
        """
        probabilities = self.predict(X_test, return_proba=True)
        predictions = (probabilities[:, 1] > threshold).astype(int)

        anomaly_indices = np.where(predictions == 1)[0]
        normal_indices = np.where(predictions == 0)[0]

        return {
            'predictions': predictions,
            'probabilities': probabilities[:, 1],
            'anomaly_indices': anomaly_indices.tolist(),
            'normal_indices': normal_indices.tolist(),
            'num_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(predictions)
        }
