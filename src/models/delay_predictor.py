"""PyTorch LSTM-based arrival delay predictor."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocess import build_delay_sequences, normalize_sequences
from src.utils.metrics import mae, rmse


class _LSTMRegressor(nn.Module):
    """Stacked LSTM with a two-layer regression head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — uses last timestep hidden state."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class DelayPredictor:
    """LSTM-based arrival delay predictor with train/predict/save/load API."""

    def __init__(self, **kwargs):
        self.cfg = {
            "sequence_length": kwargs.get("sequence_length", 10),
            "hidden_size": kwargs.get("hidden_size", 64),
            "num_layers": kwargs.get("num_layers", 2),
            "dropout": kwargs.get("dropout", 0.2),
            "epochs": kwargs.get("epochs", 15),
            "batch_size": kwargs.get("batch_size", 64),
            "lr": kwargs.get("lr", 1e-3),
        }
        self.model: _LSTMRegressor | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.input_size: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, df) -> dict:
        """Train on a flights DataFrame and return validation metrics."""
        X, y = build_delay_sequences(df, self.cfg["sequence_length"])
        if len(X) == 0:
            return {"mae": float("inf"), "rmse": float("inf")}

        X, self.mean, self.std = normalize_sequences(X)
        self.input_size = X.shape[2]

        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_dl = DataLoader(train_ds, batch_size=self.cfg["batch_size"], shuffle=True)

        self.model = _LSTMRegressor(
            self.input_size, self.cfg["hidden_size"],
            self.cfg["num_layers"], self.cfg["dropout"],
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        criterion = nn.SmoothL1Loss()

        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            running = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(xb)
            avg_loss = running / len(train_ds)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.cfg['epochs']}  loss={avg_loss:.4f}")

        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(torch.tensor(X_val).to(self.device)).cpu().numpy()
        return {
            "mae": mae(y_val, val_pred),
            "rmse": rmse(y_val, val_pred),
        }

    def predict_sequence(self, seq: np.ndarray) -> float:
        """Predict arrival delay from a single sequence (seq_len, n_features)."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        normed = (seq - self.mean) / (self.std + 1e-8)
        x = torch.tensor(normed[np.newaxis], dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x).cpu().item())

    def save(self, path: str | Path) -> None:
        """Save model state, normalization stats, and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict() if self.model else None,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "input_size": self.input_size, "cfg": self.cfg,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "DelayPredictor":
        """Load a saved DelayPredictor."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(**data["cfg"])
        obj.mean = np.array(data["mean"]) if data["mean"] is not None else None
        obj.std = np.array(data["std"]) if data["std"] is not None else None
        obj.input_size = data["input_size"]
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if data["state_dict"] is not None:
            obj.model = _LSTMRegressor(
                obj.input_size, obj.cfg["hidden_size"],
                obj.cfg["num_layers"], obj.cfg["dropout"],
            ).to(obj.device)
            obj.model.load_state_dict(data["state_dict"])
            obj.model.eval()
        return obj
