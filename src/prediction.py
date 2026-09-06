"""Stable inference API shared by Streamlit and batch consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from .data_processing import clean_dataset
    from .feature_engineering import MODEL_FEATURES, build_features
except ImportError:
    from data_processing import clean_dataset
    from feature_engineering import MODEL_FEATURES, build_features


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError("Invalid model bundle. Retrain with src/train_model.py.")
    return bundle


def _interval(prediction: float, metadata: dict[str, Any]) -> tuple[float, float]:
    q10 = float(metadata.get("prediction_interval_log_residual_q10", -0.5))
    q90 = float(metadata.get("prediction_interval_log_residual_q90", 0.5))
    lower = max(0.0, np.expm1(np.log1p(prediction) + q10))
    upper = max(lower, np.expm1(np.log1p(prediction) + q90))
    return float(lower), float(upper)


def predict_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    clean = clean_dataset(df, require_target=False)
    features = build_features(clean)[MODEL_FEATURES]
    revenue = np.clip(bundle["model"].predict(features), 0, None)
    result = clean.copy()
    result["predicted_revenue"] = revenue
    intervals = [_interval(value, bundle["metadata"]) for value in revenue]
    result["revenue_lower"] = [x[0] for x in intervals]
    result["revenue_upper"] = [x[1] for x in intervals]
    price = (result["product_price"] * (1 - result["discount_rate"])).clip(lower=1)
    result["expected_orders"] = np.rint(result["predicted_revenue"] / price).astype(int)
    result["expected_roas"] = result["predicted_revenue"] / result["creator_cost"].replace(0, np.nan)
    result["expected_roi_pct"] = (
        (result["predicted_revenue"] * result["gross_margin_rate"] - result["creator_cost"])
        / result["creator_cost"].replace(0, np.nan)
        * 100
    )
    return result


def predict_one(payload: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame([payload])
    row = predict_dataframe(frame, bundle).iloc[0]
    return row.to_dict()
