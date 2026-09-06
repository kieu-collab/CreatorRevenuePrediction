"""Generate leakage-resistant out-of-fold revenue forecasts for every KOL."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

try:
    from .data_processing import load_and_clean
    from .feature_engineering import MODEL_FEATURES, build_features
    from .prediction import load_model_bundle
    from .train_model import build_estimator, model_candidates, regression_metrics
except ImportError:  # direct execution: python src/batch_prediction.py
    from data_processing import load_and_clean
    from feature_engineering import MODEL_FEATURES, build_features
    from prediction import load_model_bundle
    from train_model import build_estimator, model_candidates, regression_metrics


def generate_oof_predictions(
    data_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
) -> dict:
    """Predict each row with a fold model that never trained on that creator."""
    data = load_and_clean(data_path, require_target=True)
    bundle = load_model_bundle(model_path)
    best_name = str(bundle["metadata"]["best_model"])
    candidates, skipped = model_candidates({best_name})
    if best_name not in candidates:
        reason = skipped.get(best_name, "unknown reason")
        raise RuntimeError(f"Best model {best_name!r} is unavailable: {reason}")

    features = build_features(data)
    X = features[MODEL_FEATURES]
    y = data["revenue"].astype(float).to_numpy()
    groups = data["creator_id"].astype(str).to_numpy()
    folds = min(n_splits, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("At least two unique creators are required for out-of-fold prediction.")

    splitter = GroupKFold(n_splits=folds)
    oof = np.full(len(data), np.nan)
    fold_id = np.full(len(data), -1, dtype=int)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        estimator = build_estimator(clone(candidates[best_name]))
        estimator.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = np.clip(estimator.predict(X.iloc[valid_idx]), 0, None)
        fold_id[valid_idx] = fold

    if np.isnan(oof).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")

    log_residual = np.log1p(y) - np.log1p(oof)
    q10, q90 = np.quantile(log_residual, [0.10, 0.90])
    lower = np.maximum(0, np.expm1(np.log1p(oof) + q10))
    upper = np.maximum(lower, np.expm1(np.log1p(oof) + q90))
    effective_price = (data["product_price"] * (1 - data["discount_rate"])).clip(lower=1)

    detail = data.copy()
    detail["creator_power_score"] = features["creator_power_score"]
    detail["creator_tier"] = features["creator_tier"]
    detail["oof_fold"] = fold_id
    detail["predicted_revenue"] = oof
    detail["revenue_lower_80"] = lower
    detail["revenue_upper_80"] = upper
    detail["expected_orders"] = np.rint(oof / effective_price).astype(int)
    detail["expected_roas"] = oof / data["creator_cost"].replace(0, np.nan)
    detail["expected_roi_pct"] = (
        (oof * data["gross_margin_rate"] - data["creator_cost"]) / data["creator_cost"].replace(0, np.nan) * 100
    )
    detail["absolute_error"] = np.abs(y - oof)
    detail["absolute_percentage_error"] = np.where(y > 0, detail["absolute_error"] / y * 100, np.nan)
    detail["prediction_scope"] = "grouped_out_of_fold_unseen_creator"

    ranking = (
        detail.groupby(["creator_id", "creator_name"], as_index=False)
        .agg(
            campaigns_observed=("brand_name", "size"),
            brands_observed=("brand_name", "nunique"),
            followers=("followers", "median"),
            avg_views=("avg_views", "median"),
            creator_power_score=("creator_power_score", "median"),
            mean_predicted_revenue=("predicted_revenue", "mean"),
            median_predicted_revenue=("predicted_revenue", "median"),
            mean_actual_revenue=("revenue", "mean"),
            total_actual_revenue=("revenue", "sum"),
            mean_expected_orders=("expected_orders", "mean"),
            mean_expected_roi_pct=("expected_roi_pct", "mean"),
            median_absolute_percentage_error=("absolute_percentage_error", "median"),
        )
        .sort_values(["mean_predicted_revenue", "creator_power_score"], ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "revenue_rank", np.arange(1, len(ranking) + 1))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "kol_revenue_predictions.csv", index=False)
    ranking.to_csv(output_dir / "kol_revenue_ranking.csv", index=False)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": best_name,
        "protocol": f"{folds}-fold GroupKFold by creator_id",
        "rows": len(detail),
        "creators": int(detail["creator_id"].nunique()),
        "metrics": regression_metrics(pd.Series(y), oof),
        "interval_log_residual_q10": float(q10),
        "interval_log_residual_q90": float(q90),
        "outputs": ["kol_revenue_predictions.csv", "kol_revenue_ranking.csv"],
        "warning": (
            "Current seed data contains same-period activity proxies and default marketing fields; "
            "use rankings as MVP evidence only."
        ),
    }
    (output_dir / "batch_prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict revenue for every KOL with grouped out-of-fold validation.")
    parser.add_argument("--data", default="data/creator_campaign.csv")
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    summary = generate_oof_predictions(args.data, args.model, args.output_dir, args.folds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
