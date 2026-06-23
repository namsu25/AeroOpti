"""XGBoost-based fuel burn predictor."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.data.preprocess import build_fuel_features
from src.utils.metrics import mae, mape, r_squared, rmse


class FuelPredictor:
    """Wrapper around XGBRegressor for segment-level fuel prediction."""

    def __init__(self, **kwargs):
        defaults = dict(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=42,
        )
        defaults.update(kwargs)
        self.model = xgb.XGBRegressor(**defaults)
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> dict:
        """Train on a flights DataFrame and return validation metrics."""
        X, y = build_fuel_features(df)
        self.feature_names = list(X.columns)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = self.model.predict(X_val)
        return {
            "rmse": rmse(y_val.values, preds),
            "mae": mae(y_val.values, preds),
            "mape": mape(y_val.values, preds),
            "r2": r_squared(y_val.values, preds),
        }

    def predict_row(
        self, aircraft: str, distance_km: float, cruise_alt_ft: float,
        payload_kg: float, headwind_kts: float, temp_dev_c: float,
        turbulence_idx: float,
    ) -> float:
        """Predict fuel_kg for a single flight segment."""
        row = {
            "distance_km": distance_km, "cruise_alt_ft": cruise_alt_ft,
            "payload_kg": payload_kg, "headwind_kts": headwind_kts,
            "temp_dev_c": temp_dev_c, "turbulence_idx": turbulence_idx,
        }
        for fn in self.feature_names:
            if fn.startswith("ac_"):
                ac_code = fn.replace("ac_", "")
                row[fn] = 1.0 if ac_code == aircraft else 0.0
        x = pd.DataFrame([row])[self.feature_names]
        return float(self.model.predict(x)[0])

    def predict_segments(self, segments_df: pd.DataFrame) -> np.ndarray:
        """Predict fuel for a DataFrame of segments (must have feature columns)."""
        return self.model.predict(segments_df[self.feature_names])

    def save(self, path: str | Path) -> None:
        """Persist model and feature names via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FuelPredictor":
        """Load a saved FuelPredictor."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        return obj
