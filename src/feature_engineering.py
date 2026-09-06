"""Leakage-safe feature engineering for pre-booking revenue forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_NUMERIC_FEATURES = [
    "followers",
    "avg_views",
    "engagement_rate",
    "product_price",
    "discount_rate",
    "campaign_duration_days",
    "planned_posts",
    "planned_live_sessions",
    "products_promoted",
    "historical_conversion_rate",
    "brand_fit_score",
]

CATEGORICAL_FEATURES = ["creator_niche", "product_category", "creator_tier", "brand_name"]

ENGINEERED_NUMERIC_FEATURES = [
    "log_followers",
    "log_avg_views",
    "log_product_price",
    "views_per_follower",
    "engaged_views",
    "discounted_price",
    "planned_reach",
    "campaign_intensity",
    "creator_power_score",
    "brand_fit_interaction",
]

MODEL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_COLUMNS = {
    "revenue",
    "gmv",
    "orders",
    "live_gmv",
    "video_gmv",
    "showcase_gmv",
    "content_efficiency",
    "creator_roi",
    "rps",
}


def creator_tier(followers: pd.Series) -> pd.Series:
    bins = [-np.inf, 10_000, 100_000, 500_000, 1_000_000, np.inf]
    labels = ["Nano", "Micro", "Mid-tier", "Macro", "Mega"]
    return pd.cut(followers, bins=bins, labels=labels, right=False).astype(str)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic marketing features available before campaign launch."""
    out = df.copy()
    for column in RAW_NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")

    out["creator_tier"] = creator_tier(out["followers"].fillna(0))
    out["log_followers"] = np.log1p(out["followers"].clip(lower=0))
    out["log_avg_views"] = np.log1p(out["avg_views"].clip(lower=0))
    out["log_product_price"] = np.log1p(out["product_price"].clip(lower=0))
    out["views_per_follower"] = (
        (out["avg_views"] / out["followers"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(upper=10)
    )
    out["engaged_views"] = out["avg_views"] * out["engagement_rate"]
    out["discounted_price"] = out["product_price"] * (1 - out["discount_rate"])
    out["planned_reach"] = out["avg_views"] * out["planned_posts"].clip(lower=0)
    out["campaign_intensity"] = (out["planned_posts"] + 2.5 * out["planned_live_sessions"]) / out[
        "campaign_duration_days"
    ].clip(lower=1)

    follower_score = np.log1p(out["followers"].clip(lower=0)) / np.log1p(10_000_000)
    views_score = np.log1p(out["avg_views"].clip(lower=0)) / np.log1p(10_000_000)
    engagement_score = (out["engagement_rate"] / 0.15).clip(0, 1)
    conversion_score = (out["historical_conversion_rate"] / 0.10).clip(0, 1)
    out["creator_power_score"] = 100 * (
        0.25 * follower_score.clip(0, 1)
        + 0.30 * views_score.clip(0, 1)
        + 0.25 * engagement_score
        + 0.20 * conversion_score
    )
    out["brand_fit_interaction"] = (out["brand_fit_score"] / 100.0) * out["engaged_views"]

    for column in CATEGORICAL_FEATURES:
        out[column] = out.get(column, "Unknown").fillna("Unknown").astype(str)
    return out


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = build_features(df)
    leaked = LEAKAGE_COLUMNS.intersection(MODEL_FEATURES)
    if leaked:
        raise RuntimeError(f"Target leakage columns entered the model: {sorted(leaked)}")
    return featured[MODEL_FEATURES]


def post_campaign_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monitoring KPIs. These are deliberately excluded from MODEL_FEATURES."""
    out = df.copy()
    out["content_efficiency"] = out["revenue"] / out["avg_views"].replace(0, np.nan)
    out["conversion_potential"] = out["orders"] / out.get("product_clicks", pd.Series(np.nan, index=out.index)).replace(
        0, np.nan
    )
    out["creator_roas"] = out["revenue"] / out["creator_cost"].replace(0, np.nan)
    out["creator_roi"] = (out["revenue"] * out["gross_margin_rate"] - out["creator_cost"]) / out[
        "creator_cost"
    ].replace(0, np.nan)
    return out
