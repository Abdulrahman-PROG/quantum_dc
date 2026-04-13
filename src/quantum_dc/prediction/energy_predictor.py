"""
Energy Consumption Prediction — LSTM-based forecaster
Predicts data center energy consumption based on workload and environment.
Falls back to Ridge regression if PyTorch is unavailable.
"""

import numpy as np
from typing import Dict

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# LSTM model definition
# ---------------------------------------------------------------------------
class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq, hidden)
        return self.fc(out[:, -1, :]).squeeze(-1)   # last time-step → scalar


# ---------------------------------------------------------------------------
# Public predictor class
# ---------------------------------------------------------------------------
class EnergyPredictor:
    """
    LSTM energy predictor.
    Input shape per sample: [workload, outdoor_temp, time_of_day]
    The LSTM sees a sliding window of SEQ_LEN steps.
    Falls back to Ridge regression when PyTorch is absent.
    """

    SEQ_LEN    = 6      # look-back window (hours)
    HIDDEN     = 64
    LAYERS     = 2
    EPOCHS     = 150
    LR         = 1e-3
    BATCH_SIZE = 32

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self._use_lstm = _TORCH_AVAILABLE

        if self._use_lstm:
            self.model = None          # built lazily after input_size is known
        else:
            self._ridge = Ridge(alpha=1.0)

    # ------------------------------------------------------------------
    def _make_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Slide a window of SEQ_LEN over X (and optionally y)."""
        n = len(X)
        Xs, ys = [], []
        for i in range(n - self.SEQ_LEN):
            Xs.append(X[i: i + self.SEQ_LEN])
            if y is not None:
                ys.append(y[i + self.SEQ_LEN])
        Xs = np.array(Xs, dtype=np.float32)
        if y is not None:
            return Xs, np.array(ys, dtype=np.float32)
        return Xs

    def _poly_features(self, X: np.ndarray) -> np.ndarray:
        """Ridge fallback feature engineering (kept for parity)."""
        w   = X[:, 0]
        t   = X[:, 1]
        tod = X[:, 2] if X.shape[1] > 2 else np.zeros(len(X))
        return np.column_stack([w, t,
                                 np.sin(2 * np.pi * tod),
                                 np.cos(2 * np.pi * tod),
                                 w ** 2, w * t,
                                 np.ones(len(X))])

    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict:
        """
        Train the energy predictor.

        Args:
            X_train: (N, 3)  columns = [workload, outdoor_temp, time_of_day]
            y_train: (N,)    energy consumption in kW
        Returns:
            dict with training metrics
        """
        if not self._use_lstm:
            Xf = self._poly_features(X_train)
            Xs = self.scaler_X.fit_transform(Xf)
            ys = y_train
            self._ridge.fit(Xs, ys)
            self.is_trained = True
            pred = self._ridge.predict(Xs)
            return self._metrics(pred, ys, method='Ridge')

        # Scale
        X_scaled = self.scaler_X.fit_transform(X_train).astype(np.float32)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)

        # Need at least SEQ_LEN+1 samples
        if len(X_scaled) <= self.SEQ_LEN:
            self._use_lstm = False
            return self.train(X_train, y_train, **kwargs)

        X_seq, y_seq = self._make_sequences(X_scaled, y_scaled)

        input_size = X_seq.shape[2]
        self.model = _LSTMNet(input_size, self.HIDDEN, self.LAYERS)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        loss_fn    = nn.MSELoss()

        Xt = torch.from_numpy(X_seq)
        yt = torch.from_numpy(y_seq)

        self.model.train()
        epochs = kwargs.get('epochs', self.EPOCHS)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.BATCH_SIZE):
                idx  = perm[i: i + self.BATCH_SIZE]
                pred = self.model(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(Xt).numpy()

        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        actual = self.scaler_y.inverse_transform(y_seq.reshape(-1, 1)).ravel()
        self.is_trained = True
        return self._metrics(pred, actual, method='LSTM')

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict energy consumption.

        Args:
            X_test: (M, 3) — workload, outdoor_temp, time_of_day
        Returns:
            (M,) predicted energy in kW
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if not self._use_lstm:
            Xf = self._poly_features(X_test)
            return self._ridge.predict(self.scaler_X.transform(Xf))

        X_scaled = self.scaler_X.transform(X_test).astype(np.float32)

        # Pad if fewer than SEQ_LEN rows
        if len(X_scaled) < self.SEQ_LEN:
            pad = np.zeros((self.SEQ_LEN - len(X_scaled), X_scaled.shape[1]), dtype=np.float32)
            X_scaled = np.vstack([pad, X_scaled])

        X_seq = self._make_sequences(X_scaled)   # no y → returns array only
        if len(X_seq) == 0:
            # single-step fallback: use last SEQ_LEN rows directly
            X_seq = X_scaled[-self.SEQ_LEN:][np.newaxis]  # (1, SEQ_LEN, features)

        Xt = torch.from_numpy(X_seq.astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(Xt).numpy()

        return self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        pred = self.predict(X_test)
        n = min(len(pred), len(y_test))
        return self._metrics(pred[:n], y_test[:n], method='LSTM' if self._use_lstm else 'Ridge')

    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(pred: np.ndarray, actual: np.ndarray, method: str) -> Dict:
        mse  = float(np.mean((pred - actual) ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(pred - actual)))
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {'method': method, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
                'predictions': pred, 'actuals': actual}

    @property
    def model_name(self) -> str:
        return 'LSTM' if (self._use_lstm and self.is_trained) else 'Ridge'
