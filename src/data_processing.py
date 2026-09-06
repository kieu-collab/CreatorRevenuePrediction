"""Load, normalize, validate and export Creator/KOL campaign data.

The module accepts either the canonical project schema or common Kalodata
exports. Raw sources are never overwritten.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

TARGET = "revenue"

CANONICAL_COLUMNS = [
    "creator_id",
    "creator_name",
    "brand_name",
    "creator_niche",
    "product_category",
    "campaign_date",
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
    "creator_cost",
    "gross_margin_rate",
    "orders",
    "gmv",
    "revenue",
]

NUMERIC_COLUMNS = [
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
    "creator_cost",
    "gross_margin_rate",
    "orders",
    "gmv",
    "revenue",
]

ALIASES = {
    "handle": "creator_id",
    "nickname": "creator_name",
    "brand_name": "brand_name",
    "shop name": "brand_name",
    "category": "product_category",
    "date range": "campaign_date",
    "followers": "followers",
    "views": "total_views",
    "avg_views": "avg_views",
    "engagement rate": "engagement_rate",
    "engagement_rate": "engagement_rate",
    "price(₫)": "product_price",
    "shop_avgprice": "product_price",
    "avg. unit price(₫)": "product_price",
    "productcount": "products_promoted",
    "videonum": "planned_posts",
    "livenum": "planned_live_sessions",
    "revenue(₫)": "revenue",
    "revenue": "revenue",
    "item sold": "orders",
    "itemsold": "orders",
}

DEFAULTS = {
    "creator_name": "Unknown creator",
    "brand_name": "Unknown brand",
    "creator_niche": "Beauty & Personal Care",
    "product_category": "Beauty & Personal Care",
    "campaign_date": pd.Timestamp("2025-10-08"),
    "engagement_rate": 0.04,
    "product_price": 250_000.0,
    "discount_rate": 0.0,
    "campaign_duration_days": 30.0,
    "planned_posts": 1.0,
    "planned_live_sessions": 0.0,
    "products_promoted": 1.0,
    "historical_conversion_rate": 0.02,
    "brand_fit_score": 70.0,
    "creator_cost": 0.0,
    "gross_margin_rate": 0.35,
}


def parse_compact_number(value: object) -> float:
    """Parse Kalodata values such as ``₫78.96b``, ``46.31k`` and ``12.5%``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower().replace(",", "")
    if text in {"", "-", "nan", "none", "n/a"}:
        return np.nan
    is_percent = text.endswith("%")
    text = text.replace("₫", "").replace("vnd", "").replace("%", "").strip()
    multiplier = 1.0
    if text.endswith("k"):
        multiplier, text = 1_000.0, text[:-1]
    elif text.endswith("m"):
        multiplier, text = 1_000_000.0, text[:-1]
    elif text.endswith("b"):
        multiplier, text = 1_000_000_000.0, text[:-1]
    try:
        result = float(text) * multiplier
        return result / 100.0 if is_percent else result
    except ValueError:
        return np.nan


def _normalize_header(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\n", " ").strip()).lower()


def _read_excel(source: str | Path | BinaryIO) -> pd.DataFrame:
    workbook = pd.ExcelFile(source)
    candidates = [s for s in workbook.sheet_names if s.lower() != "intro"]
    if not candidates:
        raise ValueError("Workbook does not contain a data sheet.")
    return pd.read_excel(source, sheet_name=candidates[0])


def load_tabular(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Read CSV/XLSX from a path or Streamlit uploaded file."""
    name = filename or getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise ValueError("Supported formats are CSV, XLSX and XLS.")


def _parse_campaign_date(series: pd.Series) -> pd.Series:
    first_date = series.astype(str).str.split("~").str[0]
    return pd.to_datetime(first_date, errors="coerce")


def normalize_columns(raw: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """Convert canonical, Kalodata Creator, or project master data to one schema."""
    if raw.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = raw.copy()
    rename = {}
    for column in df.columns:
        key = _normalize_header(column)
        rename[column] = ALIASES.get(key, key.replace(" ", "_"))
    df = df.rename(columns=rename)

    # Excel group headers are commonly represented as merged cells. Forward fill
    # only group-level shop attributes, never creator outcomes.
    for column in ["brand_name", "shop_revenue", "shop_affiliaterevenue", "shoppingmallrevenue", "product_price"]:
        if column in df:
            df[column] = df[column].ffill()

    # Preserve both canonical avg_views and Kalodata's period total Views.
    if "avg_views" not in df and "total_views" in df:
        activities = (
            pd.to_numeric(df.get("planned_posts", 0), errors="coerce").fillna(0)
            + pd.to_numeric(df.get("planned_live_sessions", 0), errors="coerce").fillna(0)
        ).clip(lower=1)
        df["avg_views"] = pd.to_numeric(df["total_views"], errors="coerce") / activities

    if "creator_id" not in df:
        df["creator_id"] = [f"creator_{i:06d}" for i in range(len(df))]
    if "creator_name" not in df:
        df["creator_name"] = df["creator_id"]

    for column, value in DEFAULTS.items():
        if column not in df:
            df[column] = value

    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = df[column].map(parse_compact_number)

    df["campaign_date"] = _parse_campaign_date(df["campaign_date"])
    df["creator_id"] = df["creator_id"].astype(str).str.strip()
    df["creator_name"] = df["creator_name"].fillna(df["creator_id"]).astype(str).str.strip()

    # Treat rates supplied as 4.5 rather than 0.045 as percentages.
    for column in ["engagement_rate", "discount_rate", "historical_conversion_rate", "gross_margin_rate"]:
        mask = df[column] > 1
        df.loc[mask, column] = df.loc[mask, column] / 100.0
        df[column] = df[column].clip(lower=0, upper=1)
    df["brand_fit_score"] = df["brand_fit_score"].clip(lower=0, upper=100)

    if "gmv" not in df:
        df["gmv"] = df.get("revenue", np.nan)
    if "orders" not in df:
        effective_price = (df["product_price"] * (1 - df["discount_rate"])).replace(0, np.nan)
        df["orders"] = np.floor(df.get("revenue", np.nan) / effective_price)

    for column in CANONICAL_COLUMNS:
        if column not in df:
            df[column] = np.nan

    df["source_file"] = source_name
    return df[CANONICAL_COLUMNS + ["source_file"]]


def clean_dataset(raw: pd.DataFrame, require_target: bool = True, source_name: str = "") -> pd.DataFrame:
    """Normalize values and remove invalid/duplicate campaign rows."""
    df = normalize_columns(raw, source_name=source_name)
    required = ["creator_id", "followers", "avg_views"] + ([TARGET] if require_target else [])
    df = df.dropna(subset=required).copy()
    df = df[(df["followers"] >= 0) & (df["avg_views"] >= 0)]
    if require_target:
        df = df[df[TARGET] >= 0]
    key = ["creator_id", "brand_name", "campaign_date"]
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    return df


def validate_dataset(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Return actionable quality warnings; an empty list means validation passed."""
    warnings: list[str] = []
    required = {"creator_id", "followers", "avg_views"}
    if require_target:
        required.add(TARGET)
    missing = sorted(required.difference(df.columns))
    if missing:
        warnings.append(f"Missing required columns: {', '.join(missing)}")
        return warnings
    if len(df) < 50:
        warnings.append("Fewer than 50 valid rows: metrics will be unstable.")
    if df["creator_id"].nunique() < 10:
        warnings.append("Fewer than 10 unique creators: creator-level holdout is not reliable.")
    if require_target and (df[TARGET] == 0).mean() > 0.3:
        warnings.append("More than 30% of target values are zero; consider a two-stage zero/non-zero model.")
    return warnings


def load_and_clean(
    source: str | Path | BinaryIO, require_target: bool = True, filename: str | None = None
) -> pd.DataFrame:
    raw = load_tabular(source, filename=filename)
    return clean_dataset(
        raw, require_target=require_target, source_name=filename or getattr(source, "name", str(source))
    )


def combine_sources(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [load_and_clean(path, filename=Path(path).name) for path in paths]
    if not frames:
        raise ValueError("No source files were supplied.")
    return pd.concat(frames, ignore_index=True)
