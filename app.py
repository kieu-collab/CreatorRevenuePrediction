from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.batch_prediction import generate_oof_predictions
from src.data_processing import load_and_clean, validate_dataset
from src.feature_engineering import build_features
from src.prediction import load_model_bundle, predict_one
from src.train_model import train_and_save

MODEL_PATH = ROOT / "models" / "best_model.pkl"
DATA_PATH = ROOT / "data" / "creator_campaign.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="Creator Revenue Prediction", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background:#f7f8fb; border:1px solid #e7e9ef; padding:16px; border-radius:12px;}
    div[data-testid="stForm"] {border:1px solid #e7e9ef; padding:18px; border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle(modified_time: float):
    del modified_time
    return load_model_bundle(MODEL_PATH)


@st.cache_data
def read_artifact(name: str) -> pd.DataFrame:
    path = MODEL_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def read_output(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def vnd(value: float) -> str:
    return f"{value:,.0f} ₫"


def model_ready() -> bool:
    return MODEL_PATH.exists()


bundle = get_bundle(MODEL_PATH.stat().st_mtime) if model_ready() else None
metadata = bundle["metadata"] if bundle else {}
defaults = metadata.get("defaults", {})

st.title("Creator Revenue Prediction")
st.caption("Dự báo doanh thu/GMV Creator tạo ra cho thương hiệu mỹ phẩm & FMCG trên TikTok Shop")

page = st.sidebar.radio(
    "Điều hướng",
    ["Creator Revenue Prediction", "Creator Analytics Dashboard", "Scenario Simulation", "Data & Model Management"],
)
if bundle:
    st.sidebar.success(f"Model: {metadata.get('best_model', 'N/A')}")
    st.sidebar.caption(f"{metadata.get('rows', 0):,} dòng · {metadata.get('unique_creators', 0):,} creators")
else:
    st.sidebar.warning("Chưa có model. Hãy train ở trang Data & Model Management.")


def input_form(prefix: str = "base") -> dict:
    with st.form(f"{prefix}_prediction_form"):
        st.subheader("Creator information")
        c1, c2, c3 = st.columns(3)
        followers = c1.number_input(
            "Followers", min_value=0, value=int(defaults.get("followers", 100_000)), step=10_000
        )
        avg_views = c2.number_input(
            "Average views/content", min_value=0, value=int(defaults.get("avg_views", 50_000)), step=5_000
        )
        engagement = c3.number_input(
            "Engagement rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.get("engagement_rate", 0.04) * 100),
            step=0.1,
        )

        st.subheader("Product & fit")
        c4, c5, c6 = st.columns(3)
        price = c4.number_input(
            "List price (VND)", min_value=1_000, value=int(defaults.get("product_price", 250_000)), step=10_000
        )
        category = c5.selectbox(
            "Category", ["Skincare", "Makeup", "Fragrance", "Personal Care", "Food & Beverage", "Household", "Other"]
        )
        brand_fit = c6.slider("Brand fit score", 0, 100, int(defaults.get("brand_fit_score", 70)))

        st.subheader("Campaign plan")
        c7, c8, c9, c10 = st.columns(4)
        duration = c7.number_input(
            "Duration (days)", min_value=1, value=int(defaults.get("campaign_duration_days", 30))
        )
        posts = c8.number_input("Planned videos", min_value=0, value=int(defaults.get("planned_posts", 3)))
        lives = c9.number_input(
            "Planned live sessions", min_value=0, value=int(defaults.get("planned_live_sessions", 1))
        )
        discount = c10.number_input(
            "Discount (%)", min_value=0.0, max_value=90.0, value=float(defaults.get("discount_rate", 0) * 100), step=1.0
        )

        c11, c12, c13 = st.columns(3)
        conversion = c11.number_input(
            "Historical conversion rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.get("historical_conversion_rate", 0.02) * 100),
            step=0.1,
        )
        creator_cost = c12.number_input(
            "Creator booking cost (VND)",
            min_value=0,
            value=int(defaults.get("creator_cost", 10_000_000)),
            step=1_000_000,
        )
        gross_margin = c13.number_input(
            "Gross margin (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.get("gross_margin_rate", 0.35) * 100),
            step=1.0,
        )
        submitted = st.form_submit_button("Predict Revenue", type="primary", width="stretch")
    return {
        "submitted": submitted,
        "creator_id": f"{prefix}_creator",
        "creator_name": "Prospective creator",
        "brand_name": "Prospective brand",
        "creator_niche": "Beauty & Personal Care",
        "product_category": category,
        "campaign_date": pd.Timestamp.today().normalize(),
        "followers": followers,
        "avg_views": avg_views,
        "engagement_rate": engagement / 100,
        "product_price": price,
        "discount_rate": discount / 100,
        "campaign_duration_days": duration,
        "planned_posts": posts,
        "planned_live_sessions": lives,
        "products_promoted": 1,
        "historical_conversion_rate": conversion / 100,
        "brand_fit_score": brand_fit,
        "creator_cost": creator_cost,
        "gross_margin_rate": gross_margin / 100,
    }


if page == "Creator Revenue Prediction":
    payload = input_form()
    if payload.pop("submitted"):
        if not bundle:
            st.error("Chưa có model để dự báo.")
        else:
            result = predict_one(payload, bundle)
            st.session_state["last_payload"] = payload
            st.session_state["last_result"] = result
            st.subheader("Prediction")
            a, b, c = st.columns(3)
            a.metric("Expected Revenue", vnd(result["predicted_revenue"]))
            b.metric("Expected Orders", f"{int(result['expected_orders']):,}")
            roi = result["expected_roi_pct"]
            c.metric("Expected ROI", "N/A" if pd.isna(roi) else f"{roi:,.1f}%")
            st.caption(
                f"Khoảng tham chiếu 80% theo residual holdout: {vnd(result['revenue_lower'])} – {vnd(result['revenue_upper'])}. "
                "Đây là estimate ra quyết định, không phải cam kết doanh thu."
            )
            features = build_features(pd.DataFrame([payload])).iloc[0]
            st.info(
                f"Creator Power Score: {features['creator_power_score']:.1f}/100 · Planned reach: {features['planned_reach']:,.0f}"
            )

elif page == "Creator Analytics Dashboard":
    st.header("Creator Analytics Dashboard")
    metrics = read_artifact("metrics.csv")
    importance = read_artifact("feature_importance.csv")
    predictions = read_artifact("test_predictions.csv")
    errors = read_artifact("error_analysis.csv")
    full_ranking = read_output("kol_revenue_ranking.csv")
    if metrics.empty:
        st.info("Train model để tạo dashboard đánh giá.")
    else:
        best = metrics.sort_values("RMSE").iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", vnd(best["MAE"]))
        m2.metric("RMSE", vnd(best["RMSE"]))
        m3.metric("MAPE", f"{best['MAPE']:.1f}%")
        m4.metric("R²", f"{best['R2']:.3f}")
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                metrics.sort_values("RMSE"), x="Model", y="RMSE", color="Model", title="Model comparison — RMSE"
            )
            st.plotly_chart(fig, width="stretch")
        with right:
            top = importance.head(12).sort_values("importance_mean")
            fig = px.bar(
                top, x="importance_mean", y="feature", orientation="h", title="Permutation importance — holdout MAE"
            )
            st.plotly_chart(fig, width="stretch")
        if not predictions.empty:
            predictions["actual_revenue"] = pd.to_numeric(predictions["actual_revenue"], errors="coerce")
            predictions["predicted_revenue"] = pd.to_numeric(predictions["predicted_revenue"], errors="coerce")
            fig = px.scatter(
                predictions,
                x="actual_revenue",
                y="predicted_revenue",
                hover_data=["creator_name", "brand_name"],
                title="Actual vs predicted revenue",
            )
            max_value = max(predictions["actual_revenue"].max(), predictions["predicted_revenue"].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_value],
                    y=[0, max_value],
                    mode="lines",
                    name="Perfect prediction",
                    line={"dash": "dash"},
                )
            )
            st.plotly_chart(fig, width="stretch")
            st.subheader("Creator ranking — grouped out-of-fold predictions")
            if full_ranking.empty:
                ranking = (
                    predictions.groupby(["creator_id", "creator_name"], as_index=False)
                    .agg(
                        mean_predicted_revenue=("predicted_revenue", "mean"),
                        mean_actual_revenue=("actual_revenue", "mean"),
                        campaigns_observed=("brand_name", "size"),
                    )
                    .sort_values("mean_predicted_revenue", ascending=False)
                    .head(25)
                )
            else:
                ranking = full_ranking.head(25)
            st.dataframe(ranking, width="stretch", hide_index=True)
        if not errors.empty:
            st.subheader("Error analysis by creator tier")
            st.dataframe(errors, width="stretch", hide_index=True)

elif page == "Scenario Simulation":
    st.header("Scenario Simulation")
    if not bundle:
        st.info("Train model trước khi chạy scenario.")
    else:
        base = st.session_state.get(
            "last_payload",
            {
                "creator_id": "scenario_creator",
                "creator_name": "Scenario creator",
                "brand_name": "Prospective brand",
                "creator_niche": "Beauty & Personal Care",
                "product_category": "Skincare",
                "campaign_date": pd.Timestamp.today(),
                "followers": defaults.get("followers", 100_000),
                "avg_views": defaults.get("avg_views", 50_000),
                "engagement_rate": defaults.get("engagement_rate", 0.04),
                "product_price": defaults.get("product_price", 250_000),
                "discount_rate": defaults.get("discount_rate", 0),
                "campaign_duration_days": defaults.get("campaign_duration_days", 30),
                "planned_posts": defaults.get("planned_posts", 3),
                "planned_live_sessions": defaults.get("planned_live_sessions", 1),
                "products_promoted": 1,
                "historical_conversion_rate": defaults.get("historical_conversion_rate", 0.02),
                "brand_fit_score": defaults.get("brand_fit_score", 70),
                "creator_cost": defaults.get("creator_cost", 10_000_000),
                "gross_margin_rate": defaults.get("gross_margin_rate", 0.35),
            },
        )
        c1, c2, c3 = st.columns(3)
        follower_change = c1.slider("Follower change", -50, 100, 20, format="%d%%")
        discount_change = c2.slider("Discount change", -20, 40, 10, format="%d pp")
        posting_change = c3.slider("Posting frequency change", -80, 200, 25, format="%d%%")
        scenario = dict(base)
        scenario["followers"] = max(0, base["followers"] * (1 + follower_change / 100))
        scenario["discount_rate"] = min(0.9, max(0, base["discount_rate"] + discount_change / 100))
        scenario["planned_posts"] = max(0, round(base["planned_posts"] * (1 + posting_change / 100)))
        baseline_result = predict_one(base, bundle)
        scenario_result = predict_one(scenario, bundle)
        delta = scenario_result["predicted_revenue"] - baseline_result["predicted_revenue"]
        a, b, c = st.columns(3)
        a.metric("Baseline revenue", vnd(baseline_result["predicted_revenue"]))
        b.metric("Scenario revenue", vnd(scenario_result["predicted_revenue"]), delta=vnd(delta))
        c.metric(
            "Scenario ROI",
            "N/A" if pd.isna(scenario_result["expected_roi_pct"]) else f"{scenario_result['expected_roi_pct']:.1f}%",
        )
        fig = px.bar(
            pd.DataFrame(
                {
                    "Scenario": ["Baseline", "Changed inputs"],
                    "Revenue": [baseline_result["predicted_revenue"], scenario_result["predicted_revenue"]],
                }
            ),
            x="Scenario",
            y="Revenue",
            color="Scenario",
            title="Revenue sensitivity",
        )
        st.plotly_chart(fig, width="stretch")
        st.warning(
            "Scenario results show model associations, not causal lift. Use A/B tests or causal methods before treating a change as guaranteed uplift."
        )

else:
    st.header("Data & Model Management")
    st.write(
        "Upload a canonical CSV/XLSX or a Kalodata Creator export, inspect the normalized rows, then retrain the model."
    )
    uploaded = st.file_uploader("Creator campaign data", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            clean = load_and_clean(uploaded, require_target=True, filename=uploaded.name)
            warnings_list = validate_dataset(clean)
            st.success(f"{len(clean):,} valid rows · {clean['creator_id'].nunique():,} creators")
            for warning in warnings_list:
                st.warning(warning)
            st.dataframe(clean.head(50), width="stretch", hide_index=True)
            if st.button("Train models with uploaded data", type="primary"):
                upload_path = ROOT / "data" / "uploaded_creator_campaign.csv"
                clean.to_csv(upload_path, index=False)
                with st.spinner("Training and evaluating candidate models..."):
                    summary = train_and_save(upload_path, MODEL_DIR)
                    generate_oof_predictions(upload_path, MODEL_PATH, OUTPUT_DIR, n_splits=5)
                get_bundle.clear()
                read_artifact.clear()
                read_output.clear()
                st.success(f"Training complete. Best model: {summary['best_model']}. Reload the app to use it.")
        except Exception as exc:
            st.error(f"Cannot process this file: {exc}")
    st.subheader("Required production data contract")
    st.code(
        "creator_id, creator_name, followers, avg_views, engagement_rate, product_price, product_category, campaign_date, campaign_duration_days, planned_posts, planned_live_sessions, historical_conversion_rate, brand_fit_score, creator_cost, revenue"
    )
    st.caption(
        "Do not use current-campaign GMV, orders, Live_GMV, Video_GMV or Revenue-derived ratios as model inputs."
    )
