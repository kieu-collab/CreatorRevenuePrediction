# Model card — MVP snapshot

## Intended use

Shortlist and compare prospective TikTok Shop creators for beauty/FMCG campaigns. The output is creator-attributed campaign GMV/revenue in VND, plus derived expected orders and ROI. Do not use the estimate as a guarantee, automated contracting decision, or substitute for commercial review.

## Training data

- 4,798 creator × brand rows converted from the local Kalodata master.
- 3,132 unique creator handles across 50 brands.
- One main observation window rather than a longitudinal history.
- Group holdout keeps every creator entirely in either train or test: 3,814 train rows and 984 test rows.
- Revenue distribution is extremely skewed: median about 24.4 million VND, 95th percentile about 574.9 million VND and maximum about 65.0 billion VND.

## Current model result

CatBoost has the lowest holdout RMSE among the candidates that ran locally.

| Model | MAE (VND) | RMSE (VND) | MAPE | R² |
|---|---:|---:|---:|---:|
| CatBoost | 221,121,492 | 2,552,040,360 | 116.3% | 0.293 |
| Gradient Boosting | 225,400,586 | 2,599,120,219 | 111.6% | 0.266 |
| Random Forest | 233,591,137 | 2,605,253,203 | 89.4% | 0.263 |
| Neural Network | 1,399,863,453 | 31,714,159,352 | 143.2% | -108.247 |
| Linear Regression | 12,390,818,647 | 269,933,874,781 | 1,382.1% | -7,913.413 |

XGBoost and LightGBM are implemented but were skipped in the local macOS training run because `libomp` was not installed. Both normally run on Linux/Streamlit Cloud. SHAP completed for the winning CatBoost model.

The exported all-KOL ranking uses a separate 5-fold `GroupKFold` run so every one of the 4,798 rows is predicted by a model that did not train on that creator. Its aggregate out-of-fold metrics are MAE 140.1 million VND, RMSE 1.459 billion VND, MAPE 96.9% and R² 0.307. These metrics remain too weak for automated booking decisions.

## Interpretation

On this holdout, the largest permutation/SHAP signals include products promoted, campaign intensity, views/engaged views, planned reach and live-session activity. Followers contributes, but ranks below several activity/reach measures. This supports the hypothesis that follower count alone is not a sufficient booking criterion.

This is descriptive model evidence, not a causal conclusion. The current seed data uses same-period views/activity as proxies, and engagement rate, conversion rate, discount and brand-fit values are defaults. The model cannot yet answer whether increasing discount or posting frequency *causes* revenue growth.

## Known limitations and required upgrade

1. Collect multiple monthly campaign cohorts with a creator snapshot timestamp before campaign start.
2. Replace default engagement, conversion, creator fee, discount, margin and brand-fit values with observed values.
3. Add product-level category and click/order funnel data.
4. Backtest by month and brand; compare unseen-creator and known-creator use cases separately.
5. Decide whether the business target is GMV, net revenue or contribution margin and train one model per definition.
6. Recalibrate prediction intervals and monitor error/drift after each data refresh.
