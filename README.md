# 🚀 Datathon VinUni 2026 - Sales Forecasting Pipeline

Đây là mã nguồn chính thức cho **Phần III: Forecasting (Dự báo Dữ liệu)** của cuộc thi Datathon VinUni 2026 do đội **The Gridbreakers** thực hiện. Dự án áp dụng mô hình máy học (Machine Learning) để dự báo **Doanh thu (Revenue)** và **Giá vốn hàng bán (COGS)** hàng ngày trong khoảng thời gian từ 01/2023 đến 07/2024.

Mô hình đã đạt **67 Điểm (RMSE/MAE)** - một kết quả xuất sắc nhờ áp dụng triết lý "Đơn giản là sức mạnh" (Occam's Razor) kết hợp với kỹ thuật Direct Forecasting (Dự báo trực tiếp).

## 🎯 Kiến trúc Pipeline (67 Điểm)

Giải pháp sử dụng kỹ thuật **TimeSeriesSplit Ensemble** nhằm tận dụng sức mạnh của 3 thuật toán Gradient Boosting hiện đại nhất, hoàn toàn loại bỏ sai số đệ quy (Recursive Error):

- **LightGBM:** Tốc độ cao, tối ưu tuyệt vời cho dữ liệu chuỗi thời gian (Time-Series) dạng bảng.
- **XGBoost:** Mô hình Boosting mạnh mẽ, độ ổn định cực cao.
- **CatBoost:** Chống overfitting vượt trội với thuật toán xử lý cây đối xứng.

**Cơ chế hoạt động:**
1. Áp dụng phép biến đổi logarit `np.log1p` lên mục tiêu (Revenue/COGS) để dập tắt nhiễu phương sai (Variance) do các dịp Sale bùng nổ gây ra.
2. Dùng K-Fold `TimeSeriesSplit(n_splits=5)` để huấn luyện độc lập cả 3 mô hình (tạo ra 15 mô hình dự báo).
3. Lấy Trung bình cộng (Simple Averaging) của tất cả các mô hình trên tập Test (tương lai) và dùng `np.expm1` để đảo ngược về giá trị thực.

## 📁 Cấu trúc Thư mục

```bash
Datathon-VinUni/
├── data/                       # Chứa file dữ liệu (sẽ được Git ignore để tối ưu dung lượng)
├── src/
│   ├── data_loader.py          # Script load 15 tập dữ liệu CSV
│   ├── feature_engineering.py  # Script tạo 10 Đặc trưng tĩnh (Calendar Features)
│   ├── models.py               # Định nghĩa kiến trúc thuật toán (LGBM, XGB, CatBoost)
│   ├── train.py                # Pipeline Huấn luyện & Lấy biểu quyết Ensemble
│   └── utils.py                # Logging, Đánh giá mô hình, SHAP plots
├── models/                     # Chứa file submission.csv và các biểu đồ phân tích
├── logs/                       # Log ghi nhận quá trình training
├── notebooks/
│   ├── 06_datathon_overclocked.ipynb  # Notebook chính thức của bản 67 điểm
│   └── EDA_Part2_TheGridbreakers.ipynb # Notebook phân tích dữ liệu (EDA)
├── requirements.txt            # Danh sách các thư viện Python
├── main.py                     # Entry point chính của dự án
└── README.md                   # Tài liệu hướng dẫn này
```

## 🛠️ Feature Engineering (Sự Tối Giản Cốt Lõi)

Mô hình **KHÔNG sử dụng dữ liệu ngoài**, **KHÔNG sử dụng đệ quy (Lags/Rolling)** để tránh tích tụ sai số. Mọi thứ được gói gọn trong **10 đặc trưng vàng**:
1. `year`, `month`, `day`, `dayofweek`, `quarter`, `dayofyear`: Các mốc thời gian cơ bản.
2. `is_weekend`: Bắt hành vi mua sắm cuối tuần.
3. `is_payday`: Bắt hành vi "bùng nổ chi tiêu" vào các ngày nhận lương (mùng 1-3, 28-31).
4. `month_sin` / `month_cos`: Chuỗi lượng giác (Cyclic Encoding) mô phỏng tính mùa vụ vòng lặp của 12 tháng.

## 🚀 Hướng dẫn Cài đặt và Sử dụng

### 1. Môi trường (Environment)

Khuyến nghị sử dụng Conda để tạo môi trường ảo:

```bash
conda create -n datathon python=3.11.7
conda activate datathon
pip install -r requirements.txt
```

### 2. Dữ liệu (Data)

Copy toàn bộ các file `.csv` của ban tổ chức cấp (ít nhất là `sales.csv` và `sample_submission.csv`) vào trong thư mục `data/` của dự án. 
*(Lưu ý: Thư mục này đã được thiết lập `.gitignore` để không bị đẩy dữ liệu lớn lên GitHub).*

### 3. Chạy Pipeline (Training & Inference)

Sử dụng file `main.py` để khởi chạy. Quá trình chạy diễn ra cực kỳ nhanh (dưới 1 phút):

```bash
python main.py --skip-tuning
```

### 4. Kết quả đầu ra (Output)

Sau khi chạy xong, các kết quả sẽ xuất hiện trong thư mục `models/`:
- `submission.csv`: File dự báo Revenue và COGS định dạng chuẩn nộp lên Kaggle.
- `forecast_revenue.png`: Biểu đồ so sánh chuỗi thời gian của tập Train và đoạn dự báo Test.

---
*Developed by The Gridbreakers for VinUni Datathon 2026 - Round 1*
