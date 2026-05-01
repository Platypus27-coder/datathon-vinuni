# Datathon VinUni 2026 - Sales Forecasting Pipeline

Đây là project forecasting cho phần dự báo doanh thu và giá vốn hàng bán theo ngày. Pipeline hiện tại tập trung vào bài toán dự báo trực tiếp từ `sales.csv` và `sample_submission.csv`, sử dụng các đặc trưng lịch đơn giản, mô hình boosting mạnh, và cơ chế ensemble theo TimeSeriesSplit.

## Tổng Quan Pipeline

Pipeline forecasting hiện tại gồm các thành phần chính:

- Dữ liệu đầu vào chính: `sales.csv`, `sample_submission.csv`.
- Đặc trưng sử dụng: 10 calendar/payday features.
- Model: LightGBM, XGBoost, CatBoost.
- Cross-validation: `TimeSeriesSplit(n_splits=5)`.
- Target transform: train trên `np.log1p(Revenue)` và `np.log1p(COGS)`, sau đó đảo ngược bằng `np.expm1`.
- Ensemble: lấy trung bình dự báo từ các model và các fold.
- Ràng buộc sau dự báo: `COGS <= Revenue`.

Pipeline không dùng lag/rolling recursive trong phiên bản chính, nhờ vậy tránh tích lũy sai số khi dự báo nhiều ngày trong tương lai.

## Cấu Trúc Thư Mục

```text
Datathon-VinUni/
├── data/                    # Chứa các file CSV đầu vào
├── src/
│   ├── data_loader.py       # Load dữ liệu cho forecasting và EDA
│   ├── feature_engineering.py # Tạo calendar/payday features
│   ├── models.py            # Định nghĩa LightGBM, XGBoost, CatBoost
│   ├── train.py             # Pipeline train, predict và export submission
│   └── utils.py             # Logging, metrics và visualization helpers
├── models/                  # Lưu submission.csv và các biểu đồ đầu ra
├── logs/                    # Lưu log quá trình chạy pipeline
├── notebooks/
│   ├── train.ipynb          # Notebook chính cho forecasting
│   └── EDA.ipynb            # Notebook phân tích dữ liệu
├── main.py                  # Entry point để chạy toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn project
```

## Feature Engineering

Pipeline chính dùng nhóm đặc trưng lịch tối giản:

- `year`, `month`, `day`, `dayofweek`
- `is_weekend`
- `dayofyear`, `quarter`
- `is_payday`
- `month_sin`, `month_cos`

Cách thiết kế này giúp notebook và code trong `src/` khớp nhau, dễ debug, và tránh phụ thuộc vào các bảng transaction nếu chưa kiểm chứng rõ hiệu quả.

## Cài Đặt Môi Trường

Khuyến nghị chạy project bằng conda env `datathon`:

```powershell
conda create -n datathon python=3.11.7
conda activate datathon
pip install -r requirements.txt
```

## Chuẩn Bị Dữ Liệu

Đặt các file CSV vào thư mục `data/`. Tối thiểu pipeline forecasting cần:

```text
data/sales.csv
data/sample_submission.csv
```

Các file dữ liệu khác có thể giữ trong `data/` để phục vụ EDA hoặc thử nghiệm mở rộng, nhưng pipeline forecasting mặc định chỉ cần hai file trên.

## Cách Chạy Pipeline

Chạy từ thư mục gốc project:

```powershell
python main.py
```
## Kết Quả Đầu Ra

Sau khi chạy xong, kết quả chính nằm trong:

```text
models/submission.csv
```

Pipeline cũng có thể sinh thêm các biểu đồ phân tích trong `models/`, ví dụ:

```text
models/forecast_revenue.png
models/forecast_cogs.png
models/shap_revenue_drivers.png
models/shap_cogs_drivers.png
```

File `submission.csv` gồm các cột dự báo cần thiết theo định dạng của `sample_submission.csv`, đồng thời được kiểm tra để không có giá trị thiếu, không âm, và đảm bảo `COGS` không vượt quá `Revenue`.

## Notebook

- `notebooks/train.ipynb`: phiên bản notebook tự chạy độc lập của pipeline forecasting.
- `notebooks/EDA.ipynb`: notebook phân tích dữ liệu, dùng để hiểu dữ liệu và kiểm tra các giả thuyết trước khi đưa vào pipeline chính.

Khi thay đổi logic model hoặc feature trong notebook, nên đồng bộ lại với `src/` để tránh notebook và CLI tạo ra kết quả khác nhau.

## Ghi Chú Phát Triển

- Giữ pipeline mặc định đơn giản và có thể tái lập.
- Giữ README, notebook markdown và log chính thức ở dạng gọn gàng, dễ đọc.
- Nếu thêm feature mới, nên thử từng nhóm nhỏ và so sánh bằng validation trước khi đưa vào pipeline chính.
