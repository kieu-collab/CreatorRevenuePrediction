import numpy as np
import pandas as pd

from src.data_processing import clean_dataset, parse_compact_number
from src.feature_engineering import LEAKAGE_COLUMNS, MODEL_FEATURES, build_features
from src.train_model import model_filename, regression_metrics, split_data


def sample_frame(rows=60):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "creator_id": [f"c{i // 2}" for i in range(rows)],
            "creator_name": [f"Creator {i // 2}" for i in range(rows)],
            "brand_name": [f"Brand {i % 4}" for i in range(rows)],
            "creator_niche": "Beauty",
            "product_category": "Skincare",
            "campaign_date": pd.date_range("2025-01-01", periods=rows),
            "followers": rng.integers(1_000, 1_000_000, rows),
            "avg_views": rng.integers(500, 500_000, rows),
            "engagement_rate": 0.04,
            "product_price": 250_000,
            "discount_rate": 0.1,
            "campaign_duration_days": 30,
            "planned_posts": 3,
            "planned_live_sessions": 1,
            "products_promoted": 2,
            "historical_conversion_rate": 0.02,
            "brand_fit_score": 75,
            "creator_cost": 10_000_000,
            "gross_margin_rate": 0.35,
            "revenue": rng.uniform(1_000_000, 100_000_000, rows),
        }
    )


def test_compact_number_parser():
    assert parse_compact_number("₫78.96b") == 78_960_000_000
    assert parse_compact_number("46.31k") == 46_310
    assert parse_compact_number("4.5%") == 0.045


def test_features_exclude_outcomes():
    clean = clean_dataset(sample_frame())
    featured = build_features(clean)
    assert set(MODEL_FEATURES).issubset(featured.columns)
    assert not LEAKAGE_COLUMNS.intersection(MODEL_FEATURES)
    assert featured["creator_power_score"].between(0, 100).all()


def test_creator_groups_do_not_cross_split():
    clean = clean_dataset(sample_frame())
    train_idx, test_idx, strategy = split_data(clean)
    assert strategy == "group_holdout_by_creator"
    assert set(clean.iloc[train_idx].creator_id).isdisjoint(set(clean.iloc[test_idx].creator_id))


def test_regression_metrics():
    metrics = regression_metrics(pd.Series([100, 200]), np.array([110, 190]))
    assert metrics["MAE"] == 10
    assert metrics["RMSE"] == 10
    assert metrics["R2"] > 0.9


def test_candidate_model_filename_is_stable():
    assert model_filename("Linear Regression") == "model_linear_regression.pkl"
    assert model_filename("CatBoost") == "model_catboost.pkl"
