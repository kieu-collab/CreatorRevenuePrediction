# Phân tích project VN30 và kế hoạch chuyển đổi

## Kiến trúc project cũ

Repository nguồn gồm bốn lớp chính:

1. `Data Crawling/`: Selenium lấy lịch sử giao dịch từng mã từ Vietstock. `run_crawler.py` giữ cấu hình 30 ticker và khoảng ngày; `crawl_factory.py` điều khiển trình duyệt, bấm “load more”, trích bảng 18 cột và lưu một CSV cho mỗi ticker.
2. `Dataset/`: 30 CSV lịch sử và `Dataset/results/` chứa `predictions.csv`, `predictions.parquet`, `metrics.csv`.
3. `Model/pipeline.py`: job offline dùng PySpark. Dữ liệu được parse ngày/số, sắp theo ticker và thời gian; tạo lag 1/2/3/5/10, rolling mean/std 5/10/20 kết thúc tại t-1, day-of-week và month. Mỗi ticker được split 80/20 theo thứ tự thời gian. Ba mô hình MLlib (Linear Regression, Random Forest, GBT) và hai mô hình thống kê (ARIMA, Holt-Winters) tạo dự báo. MAE, RMSE, MAPE và R² được lưu theo ticker × model.
4. `app.py`: Streamlit chỉ đọc raw CSV và artifacts đã tạo; không chạy PySpark khi serving. App có ba tab: phân tích một ticker, so sánh 30 ticker, và upload CSV để phân tích nhanh bằng pandas.

Điểm tốt cần giữ: ranh giới offline training/online serving; feature lag không nhìn tương lai; split theo thời gian; nhiều baseline; cùng một bộ metrics; artifacts đơn giản để dashboard đọc.

Điểm cần sửa khi chuyển bài toán: đường dẫn Python cứng trong `pipeline.py`; toàn bộ logic nằm trong hai file lớn; chưa lưu fitted model/preprocessor nên app cũ không thực hiện inference thật cho một input mới; upload tab dùng hai model khác pipeline chính; không có test; không có model/version metadata; chưa có SHAP/feature importance; và model selection chỉ dựa trên một holdout.

Một sai lệch đánh giá đáng chú ý: các feature lag của ba mô hình ML được tạo trên toàn chuỗi trước khi lọc train/test. Vì vậy dự báo ngày thứ hai trở đi trong test dùng giá thực tế của các ngày test trước đó. Đây là thiết lập one-step-ahead có cập nhật actual, trong khi ARIMA/Holt-Winters forecast toàn bộ test horizon không nhận actual mới. Biểu đồ đặt năm model cạnh nhau nhưng protocol không hoàn toàn tương đương; kết quả trung bình trong artifact cũ là Linear Regression RMSE khoảng 1.581 và thắng cả 30 ticker, trong khi ARIMA/Holt-Winters có RMSE khoảng 21.380/24.351. Khi tái sử dụng project cũ cần xác định rõ bài toán one-step-ahead hay recursive multi-step và đánh giá tất cả model cùng protocol.

## Mapping sang Creator Revenue

| Project VN30 | Project Creator Revenue |
|---|---|
| Ticker | Creator ID + brand/product/campaign |
| Ngày giao dịch | Campaign date / snapshot date |
| Close price | Creator-attributed revenue/GMV (VND) |
| Lag/rolling price | Historical creator metrics calculated before campaign |
| 80/20 chronological split | Group holdout by creator; chronological holdout khi có nhiều kỳ |
| PySpark feature pipeline | scikit-learn `ColumnTransformer` đóng gói trong model |
| predictions + metrics | model bundle + metadata + metrics + prediction/error/explanation artifacts |
| Ticker dashboard | Prediction, creator analytics, scenario simulation |

## Giữ, thay đổi, tạo mới

**Giữ theo pattern, không copy nguyên nội dung**

- Offline pipeline tạo artifacts trước khi deploy.
- `requirements.txt`, `README.md`, `.gitignore` và một entry point Streamlit.
- MAE, RMSE, MAPE, R²; biểu đồ actual vs predicted; bảng so sánh model.

**Thay đổi hoàn toàn**

- `Data Crawling/`: thay bằng adapter đọc Kalodata CSV/XLSX và về sau là API/ELT connector có kiểm soát.
- `Dataset/`: thay 30 file ticker bằng raw/processed campaign tables và data contract.
- `Model/pipeline.py`: tách thành `data_processing.py`, `feature_engineering.py`, `train_model.py`, `prediction.py`.
- `app.py`: thay EDA chuỗi thời gian bằng prediction, ranking, model explainability và scenario.

**Tạo mới**

- Model bundle chứa preprocessor + regressor, metadata/version, residual interval.
- Database schema creators/products/brands/campaigns/fact creator-campaign.
- Leakage guard, group holdout, feature importance, SHAP summary và error analysis.
- Automated tests, sample data builder và deployment configuration.

## Cảnh báo dữ liệu hiện có

`Master_data_forML.xlsx` có 4.798 dòng và 3.132 creator. `Revenue` bằng tổng `Live_GMV + Video_GMV + Showcase_GMV` ở phần lớn dữ liệu; `RPS` bằng `Revenue / Followers`. Vì vậy các cột này không được dùng làm input dự báo trước booking. `Views`, `VideoNum` và `LiveNum` trong snapshot hiện tại cũng là số đo cùng kỳ với revenue; bản MVP dùng chúng như proxy, nhưng production phải thay bằng số liệu lịch sử chốt trước ngày bắt đầu chiến dịch.
