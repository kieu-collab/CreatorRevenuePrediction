"""Train, compare, explain and persist Creator Revenue regression models."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_processing import TARGET, load_and_clean, validate_dataset
    from .feature_engineering import CATEGORICAL_FEATURES, MODEL_FEATURES, build_features
except ImportError:  # direct execution: python src/train_model.py
    from data_processing import TARGET, load_and_clean, validate_dataset
    from feature_engineering import CATEGORICAL_FEATURES, MODEL_FEATURES, build_features


SEED = 42
NUMERIC_FEATURES = [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    non_zero = np.abs(y) > 1e-9
    mape = np.mean(np.abs((y[non_zero] - pred[non_zero]) / y[non_zero])) * 100 if non_zero.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE": float(mape),
        "R2": float(r2_score(y, pred)),
    }


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_candidates(selected: set[str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Return installed candidates and reasons for any optional model skipped."""
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.8, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=3, loss="huber", random_state=SEED
        ),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=SEED,
        ),
    }
    skipped: dict[str, str] = {}
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=SEED,
        )
    except Exception as exc:
        skipped["XGBoost"] = f"Unavailable in this runtime: {type(exc).__name__}: {exc}"
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=450,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            verbosity=-1,
            random_state=SEED,
        )
    except Exception as exc:
        skipped["LightGBM"] = f"Unavailable in this runtime: {type(exc).__name__}: {exc}"
    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=450,
            learning_rate=0.04,
            depth=6,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            random_seed=SEED,
        )
    except Exception as exc:
        skipped["CatBoost"] = f"Unavailable in this runtime: {type(exc).__name__}: {exc}"
    if selected:
        models = {name: model for name, model in models.items() if name in selected}
    return models, skipped


def build_estimator(regressor: Any) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [
            ("preprocessor", _preprocessor()),
            ("regressor", regressor),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


def split_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer an unseen-creator holdout; fall back to chronological or random split."""
    groups = df["creator_id"].astype(str)
    if groups.nunique() >= 10 and groups.duplicated().any():
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return train_idx, test_idx, "group_holdout_by_creator"
    dates = pd.to_datetime(df["campaign_date"], errors="coerce")
    if dates.notna().sum() >= 50 and dates.nunique() >= 5:
        order = np.argsort(dates.fillna(dates.min()).to_numpy())
        cut = int(len(order) * 0.8)
        return order[:cut], order[cut:], "chronological_holdout"
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    return np.asarray(train_idx), np.asarray(test_idx), "random_holdout"


def _save_shap_summary(model: TransformedTargetRegressor, X_test: pd.DataFrame, output: Path) -> str:
    try:
        import shap

        fitted_pipeline = model.regressor_
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        regressor = fitted_pipeline.named_steps["regressor"]
        transformed = preprocessor.transform(X_test.iloc[: min(250, len(X_test))])
        names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(regressor)
        values = explainer.shap_values(transformed)
        summary = pd.DataFrame(
            {
                "feature": names,
                "mean_abs_shap_log_revenue": np.abs(np.asarray(values)).mean(axis=0),
            }
        ).sort_values("mean_abs_shap_log_revenue", ascending=False)
        summary.to_csv(output, index=False)
        return "created"
    except Exception as exc:  # SHAP support varies by winning estimator/version
        pd.DataFrame(columns=["feature", "mean_abs_shap_log_revenue"]).to_csv(output, index=False)
        return f"unavailable: {type(exc).__name__}: {exc}"


def train_and_save(
    data_path: str | Path,
    model_dir: str | Path,
    selected_models: set[str] | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    clean = load_and_clean(data_path, require_target=True)
    warnings_list = validate_dataset(clean, require_target=True)
    featured = build_features(clean)
    X = featured[MODEL_FEATURES]
    y = clean[TARGET].astype(float)
    train_idx, test_idx, split_strategy = split_data(clean)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates, skipped = model_candidates(selected_models)
    if not candidates:
        raise RuntimeError("No requested model implementation is installed.")

    fitted: dict[str, TransformedTargetRegressor] = {}
    rows = []
    failures: dict[str, str] = {}
    for name, regressor in candidates.items():
        try:
            estimator = build_estimator(regressor)
            estimator.fit(X_train, y_train)
            pred = np.clip(estimator.predict(X_test), 0, None)
            rows.append({"Model": name, **regression_metrics(y_test, pred)})
            fitted[name] = estimator
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    if not rows:
        raise RuntimeError(f"All models failed: {failures}")
    metrics = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["Model"])
    best_model = fitted[best_name]
    best_pred = np.clip(best_model.predict(X_test), 0, None)

    residual_log = np.log1p(y_test.to_numpy()) - np.log1p(best_pred)
    q10, q90 = np.quantile(residual_log, [0.10, 0.90])

    prediction_rows = clean.iloc[test_idx][
        ["creator_id", "creator_name", "brand_name", "product_category", "campaign_date", "followers"]
    ].copy()
    prediction_rows["actual_revenue"] = y_test.to_numpy()
    prediction_rows["predicted_revenue"] = best_pred
    prediction_rows["absolute_error"] = np.abs(y_test.to_numpy() - best_pred)
    prediction_rows["absolute_percentage_error"] = np.where(
        y_test.to_numpy() > 0,
        prediction_rows["absolute_error"] / y_test.to_numpy() * 100,
        np.nan,
    )
    prediction_rows.to_csv(model_dir / "test_predictions.csv", index=False)

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=4,
        random_state=SEED,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(model_dir / "feature_importance.csv", index=False)

    error_analysis = (
        prediction_rows.assign(
            creator_tier=pd.cut(
                prediction_rows["followers"],
                bins=[-np.inf, 10_000, 100_000, 500_000, 1_000_000, np.inf],
                labels=["Nano", "Micro", "Mid-tier", "Macro", "Mega"],
                right=False,
            )
        )
        .groupby("creator_tier", observed=False)
        .agg(
            rows=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            MdAPE=("absolute_percentage_error", "median"),
        )
        .reset_index()
    )
    error_analysis.to_csv(model_dir / "error_analysis.csv", index=False)
    metrics.to_csv(model_dir / "metrics.csv", index=False)

    shap_status = _save_shap_summary(best_model, X_test, model_dir / "shap_summary.csv")
    defaults = {}
    for feature in [c for c in MODEL_FEATURES if c in clean.columns and c not in CATEGORICAL_FEATURES]:
        defaults[feature] = float(pd.to_numeric(clean[feature], errors="coerce").median())
    defaults.update(
        {
            "creator_niche": str(clean["creator_niche"].mode().iloc[0]),
            "product_category": str(clean["product_category"].mode().iloc[0]),
            "brand_name": "Prospective brand",
            "creator_cost": float(clean["creator_cost"].median()),
            "gross_margin_rate": float(clean["gross_margin_rate"].median()),
        }
    )

    metadata = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "rows": len(clean),
        "unique_creators": int(clean["creator_id"].nunique()),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "split_strategy": split_strategy,
        "target": TARGET,
        "target_definition": "Creator-attributed TikTok Shop GMV/revenue in VND for the campaign window.",
        "best_model": best_name,
        "features": MODEL_FEATURES,
        "defaults": defaults,
        "prediction_interval_log_residual_q10": float(q10),
        "prediction_interval_log_residual_q90": float(q90),
        "validation_warnings": warnings_list,
        "skipped_optional_models": skipped,
        "failed_models": failures,
        "shap_status": shap_status,
        "data_limitations": [
            "Current seed data is a one-period Kalodata snapshot, not a longitudinal campaign table.",
            "Engagement, conversion, discount, creator cost and brand fit are defaults unless supplied by the user.",
            "Kalodata Views and activity counts are period observations and should be replaced by lagged pre-campaign history in production.",
        ],
    }
    bundle = {"model": best_model, "metadata": metadata}
    joblib.dump(bundle, model_dir / "best_model.pkl")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Creator Revenue models.")
    parser.add_argument("--data", default="data/creator_campaign.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--models", nargs="*", help="Optional exact model names to run")
    args = parser.parse_args()
    metadata = train_and_save(args.data, args.model_dir, set(args.models) if args.models else None)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
