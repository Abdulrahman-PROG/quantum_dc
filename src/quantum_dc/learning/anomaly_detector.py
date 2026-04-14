"""
Anomaly Detection for Data Center Operations
Classical RBF-SVM  vs  Quantum Kernel SVM — side-by-side comparison.
"""

import numpy as np
from typing import Dict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit.primitives import StatevectorSampler
    _QML_AVAILABLE = True
except ImportError:
    _QML_AVAILABLE = False


class AnomalyDetector:
    """
    Dual anomaly detector:
      - Classical: RBF-SVM (sklearn)
      - Quantum:   ZZFeatureMap + FidelityQuantumKernel (Qiskit ML)
    Both are trained together; comparison metrics are stored.
    """

    def __init__(self, kernel: str = 'rbf', n_qubits: int = 3):
        self.kernel = kernel
        self.n_qubits = n_qubits          # features fed to quantum kernel
        self.scaler = StandardScaler()
        self.classical_model = None
        self.quantum_model = None
        self.is_trained = False
        self.comparison: Dict = {}        # classical vs quantum metrics

    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict:
        """
        Train both classical and quantum SVM.
        Returns classical training metrics (primary) and stores comparison.
        """
        X_scaled = self.scaler.fit_transform(X_train)

        # --- Classical SVM ---
        self.classical_model = SVC(
            kernel=self.kernel,
            gamma=kwargs.get('gamma', 'auto'),
            C=kwargs.get('C', 1.0),
            probability=True
        )
        self.classical_model.fit(X_scaled, y_train)
        classical_metrics = self._eval_metrics(
            self.classical_model.predict(X_scaled), y_train, label='Classical SVM'
        )

        # --- Quantum Kernel SVM ---
        quantum_metrics: Dict = {}
        if _QML_AVAILABLE:
            try:
                # Reduce features to n_qubits dimensions via first n columns
                n = min(self.n_qubits, X_scaled.shape[1])
                X_q = X_scaled[:, :n].astype(np.float32)

                feature_map = ZZFeatureMap(feature_dimension=n, reps=1)
                sampler = StatevectorSampler()
                fidelity = ComputeUncompute(sampler=sampler)
                qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

                # Limit training set size for speed (quantum kernel is O(n²))
                max_q = min(80, len(X_q))
                idx = np.random.default_rng(42).choice(len(X_q), max_q, replace=False)
                X_qsub, y_qsub = X_q[idx], y_train[idx]

                self.quantum_model = SVC(kernel=qkernel.evaluate, probability=True)
                self.quantum_model.fit(X_qsub, y_qsub)
                quantum_metrics = self._eval_metrics(
                    self.quantum_model.predict(X_qsub), y_qsub, label='Quantum Kernel SVM'
                )
            except Exception as e:
                quantum_metrics = {'error': str(e)[:120]}

        self.is_trained = True
        self.comparison = {
            'classical': classical_metrics,
            'quantum': quantum_metrics,
            'quantum_available': _QML_AVAILABLE,
        }

        # Return classical metrics as primary (used by existing callers)
        return classical_metrics

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray, return_proba: bool = False) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X_test)
        if return_proba:
            return self.classical_model.predict_proba(X_scaled)
        return self.classical_model.predict(X_scaled)

    def detect(self, X_test: np.ndarray, threshold: float = 0.5) -> Dict:
        proba = self.predict(X_test, return_proba=True)
        preds = (proba[:, 1] > threshold).astype(int)
        anomaly_idx = np.where(preds == 1)[0]
        normal_idx  = np.where(preds == 0)[0]
        return {
            'predictions': preds,
            'probabilities': proba[:, 1],
            'anomaly_indices': anomaly_idx.tolist(),
            'normal_indices':  normal_idx.tolist(),
            'num_anomalies': len(anomaly_idx),
            'anomaly_rate': len(anomaly_idx) / len(preds),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_metrics(y_pred: np.ndarray, y_true: np.ndarray, label: str) -> Dict:
        accuracy  = float(np.mean(y_pred == y_true))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'label': label,
            'accuracy':  round(accuracy,  4),
            'precision': round(precision, 4),
            'recall':    round(recall,    4),
            'f1_score':  round(f1,        4),
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        }
