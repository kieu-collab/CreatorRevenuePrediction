# Creator Revenue Prediction

Ứng dụng Machine Learning dự báo doanh thu/GMV kỳ vọng mà một Creator/KOL có thể tạo cho thương hiệu mỹ phẩm hoặc FMCG trong chiến dịch TikTok Shop. Công cụ hỗ trợ shortlist creator trước booking, so sánh kịch bản và giải thích các yếu tố liên quan đến dự báo.

## Business problem

Brand thường chọn creator dựa trên follower hoặc cảm tính. Project chuyển quyết định này thành bài toán regression: tại thời điểm trước campaign, dùng hồ sơ creator, lịch sử hiệu suất đã chốt, thông tin sản phẩm, độ phù hợp và kế hoạch nội dung để ước tính `revenue` theo VND.

Model là công cụ hỗ trợ quyết định, không phải cam kết doanh thu. Scenario simulation thể hiện quan hệ model học được, không chứng minh tác động nhân quả.

## Project structure

```text
CreatorRevenuePrediction/
├── data/
│   ├── creator_campaign.csv
│   └── raw/                     # local raw exports, ignored by Git
├── models/
│   ├── best_model.pkl           # preprocessor + winning regressor
│   ├── metadata.json
│   ├── metrics.csv
│   ├── feature_importance.csv
│   ├── shap_summary.csv
│   ├── test_predictions.csv
│   └── error_analysis.csv
├── outputs/
│   ├── kol_revenue_predictions.csv
│   ├── kol_revenue_ranking.csv
│   └── batch_prediction_summary.json
├── notebooks/analysis.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── prediction.py
│   └── batch_prediction.py
├── docs/
│   ├── OLD_PROJECT_ANALYSIS.md
│   ├── MODEL_CARD.md
│   └── database_schema.sql
├── scripts/build_seed_data.py
├── tests/test_pipeline.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset

Canonical grain: one row per `creator_id × brand/product × campaign`.

Required for training: `creator_id`, `followers`, `avg_views`, `revenue`. Recommended fields:

| Group | Columns |
|---|---|
| Creator | `creator_name`, `followers`, `avg_views`, `engagement_rate`, `creator_niche`, `historical_conversion_rate` |
| Product/brand | `brand_name`, `product_price`, `product_category`, `brand_fit_score`, `gross_margin_rate` |
| Campaign plan | `campaign_date`, `campaign_duration_days`, `planned_posts`, `planned_live_sessions`, `products_promoted`, `discount_rate`, `creator_cost` |
| Outcomes | `orders`, `gmv`, `revenue` |

`src/data_processing.py` đọc CSV/XLSX canonical và các export Kalodata phổ biến. Giá trị như `₫78.96b`, `46.31k`, `4.5%` được chuẩn hóa tự động.

### Leakage policy

`Live_GMV`, `Video_GMV`, `Showcase_GMV`, current-campaign orders, `RPS = Revenue/Followers`, content efficiency và ROI không được dùng làm feature. Chúng chứa hoặc được tính từ target. Production phải dùng snapshot creator có `snapshot_date < campaign_start_date`.

## Methodology

```text
Raw Kalodata / canonical upload
  → normalize + validate + preserve raw
  → leakage-safe feature engineering
  → feature selection by explicit allow-list
  → unseen-creator holdout (or chronological holdout for longitudinal data)
  → train candidate regressors with log1p target
  → evaluate MAE, RMSE, MAPE, R²
  → select lowest holdout RMSE
  → persist model + metadata + explanations
  → Streamlit inference and simulation
```

Feature engineering includes log scale, creator tier, views/follower, engaged views, discounted price, planned reach, campaign intensity, Creator Power Score and brand-fit interaction. Post-campaign Content Efficiency and Creator ROI are monitoring outputs only.

## Machine Learning approach

| Model | Why/strength | Limitation / when to use |
|---|---|---|
| Linear Regression | Fast, transparent baseline | Misses nonlinear interactions; use as sanity check |
| Random Forest | Robust nonlinear baseline, little tuning | Large model, weak extrapolation |
| Gradient Boosting | Strong on medium tabular data, Huber loss reduces outlier impact | Sequential and more sensitive to tuning |
| XGBoost | Regularized boosting, strong accuracy and SHAP support | Extra dependency and tuning |
| LightGBM | Fast on larger tables and high-dimensional one-hot data | Can overfit small datasets |
| CatBoost | Strong for nonlinear tabular relationships | Heavier dependency; current pipeline one-hot encodes for a common contract |
| Neural Network | Captures complex smooth interactions | Needs more data, scaling and careful validation; not default for small campaign tables |

All models receive the same preprocessing contract. Target uses `log1p`/`expm1` because revenue is heavily right-skewed. The winner is selected by holdout RMSE; MAE, MAPE and R² remain visible to avoid one-metric optimization.

Feature importance uses permutation importance on untouched holdout rows. SHAP is generated when the winning estimator/version supports tree explanations. Error analysis is segmented by creator tier.

## Installation

```bash
cd CreatorRevenuePrediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Build the canonical data file

For a canonical/Kalodata file:

```bash
python scripts/build_seed_data.py --input raw.xlsx --output data/creator_campaign.csv
```

The included `data/creator_campaign.csv` is an MVP conversion of the local Kalodata master. Several unavailable fields use documented defaults; replace these with true pre-campaign values before production use.

## Train

```bash
python -m src.train_model --data data/creator_campaign.csv --model-dir models
```

Optional fast smoke run:

```bash
python -m src.train_model --data data/creator_campaign.csv --model-dir models --models "Linear Regression" "Random Forest"
```

## Run Streamlit

```bash
streamlit run app.py
```

## Predict revenue for every KOL

Generate a creator-grouped out-of-fold prediction for every row and a creator-level ranking:

```bash
python -m src.batch_prediction \
  --data data/creator_campaign.csv \
  --model models/best_model.pkl \
  --output-dir outputs \
  --folds 5
```

`outputs/kol_revenue_predictions.csv` contains one KOL × brand/campaign forecast. `outputs/kol_revenue_ranking.csv` aggregates by KOL using mean/median forecast, rather than summing more rows for creators that appear with more brands. Every row is scored by a fold that did not train on that creator.

The fourth app page accepts CSV/XLSX, previews normalized data and retrains artifacts. For repeatable production jobs, prefer the CLI command and version the data/model metadata.

## Example prediction

```python
from src.prediction import load_model_bundle, predict_one

bundle = load_model_bundle("models/best_model.pkl")
result = predict_one({
    "creator_id": "creator_001",
    "creator_name": "Beauty Creator",
    "brand_name": "Example Brand",
    "creator_niche": "Skincare",
    "product_category": "Skincare",
    "campaign_date": "2026-09-15",
    "followers": 180_000,
    "avg_views": 65_000,
    "engagement_rate": 0.055,
    "product_price": 329_000,
    "discount_rate": 0.10,
    "campaign_duration_days": 14,
    "planned_posts": 4,
    "planned_live_sessions": 1,
    "products_promoted": 2,
    "historical_conversion_rate": 0.025,
    "brand_fit_score": 85,
    "creator_cost": 15_000_000,
    "gross_margin_rate": 0.40,
}, bundle)
print(result["predicted_revenue"], result["expected_orders"], result["expected_roi_pct"])
```

Expected orders are `predicted_revenue / discounted_price`. ROI is `(predicted_revenue × gross_margin_rate − creator_cost) / creator_cost`; ROAS is also returned.

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub, including the trained `models/best_model.pkl` and CSV explanation artifacts. The included model is small enough for normal Git; use object storage plus a checksum when future artifacts become large.
2. In Streamlit Community Cloud choose the repository, branch and `app.py`.
3. Use Python 3.12 and let Cloud install `requirements.txt`.
4. Do not train on every page load. Train offline, then deploy immutable artifacts. If model size becomes large, store it in an object store and verify its checksum before loading.

## Next production steps

- Collect repeated campaigns across multiple dates and attach creator snapshots from before each campaign.
- Add real product clicks, engagement, creator fee, discount, margin and creator–category labels.
- Backtest by month and brand, tune with grouped cross-validation, monitor data drift and prediction error.
- Separate GMV, net revenue and contribution margin targets; agree one business definition before model acceptance.
- Add authentication and database/object storage before exposing uploads beyond a trusted internal team.
