# Local Server AI Strategy - Support Vector Machine

## 1. Giới thiệu

Đây là project Machine Learning xây dựng hệ thống **phân loại chất lượng nông sản** bằng thuật toán **Support Vector Machine (SVM)**.

Hệ thống thực hiện toàn bộ quy trình:

```text
Dataset
   ↓
EDA
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Train SVM Model
   ↓
Evaluate
   ↓
Tune Model
   ↓
Prediction
   ↓
FastAPI
   ↓
Docker
   ↓
Ngrok
   ↓
Internet / Mobile App
```

Mục tiêu của project là xây dựng một mô hình Machine Learning đơn giản, sau đó triển khai mô hình thành API để có thể sử dụng từ máy tính hoặc ứng dụng mobile.

---

## 2. Công nghệ sử dụng

| Công nghệ             | Mục đích                 |
| --------------------- | ------------------------ |
| Python                | Ngôn ngữ lập trình       |
| NumPy                 | Tính toán dữ liệu        |
| Scikit-learn          | Machine Learning         |
| SVM / SVC             | Phân loại chất lượng     |
| Joblib                | Lưu và tải model         |
| FastAPI               | Xây dựng REST API        |
| Uvicorn               | Chạy FastAPI             |
| Pytest                | Kiểm thử                 |
| Docker                | Đóng gói và chạy API     |
| Ngrok                 | Public API ra Internet   |
| Kaggle / Google Colab | Dataset và thử nghiệm ML |

---

## 3. Dataset

Dataset sử dụng:

```text
data/agricultural_product_quality.csv
```

Dataset gồm **1.500 mẫu dữ liệu**.

### Các thuộc tính đầu vào

| Feature       | Ý nghĩa      |
| ------------- | ------------ |
| `weight`      | Khối lượng   |
| `size`        | Kích thước   |
| `moisture`    | Độ ẩm        |
| `sugar_level` | Độ đường     |
| `firmness`    | Độ cứng      |
| `color_score` | Điểm màu sắc |

### Nhãn

```text
quality
```

Có 3 mức chất lượng:

```text
0
1
2
```

Mỗi lớp có 500 mẫu.

---

## 4. Cấu trúc project

```text
Mechine-Learning/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── modeling.py
│
├── data/
│   └── agricultural_product_quality.csv
│
├── artifacts/
│   ├── eda_report.json
│   ├── metrics.json
│   └── svm_model.joblib
│
├── scripts/
│   ├── eda.py
│   └── train_model.py
│
├── tests/
│   └── ...
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 5. Machine Learning Pipeline

## Bước 1: Dataset

Đọc dữ liệu từ:

```text
data/agricultural_product_quality.csv
```

Các feature sử dụng:

```python
[
    "weight",
    "size",
    "moisture",
    "sugar_level",
    "firmness",
    "color_score"
]
```

Target:

```python
quality
```

---

## Bước 2: EDA

Chạy:

```bash
py scripts\eda.py
```

EDA kiểm tra:

* Số lượng dữ liệu
* Kiểu dữ liệu
* Missing values
* Phân bố các lớp
* Thống kê các feature
* Một số thông tin cơ bản của dataset

Kết quả được lưu tại:

```text
artifacts/eda_report.json
```

---

## Bước 3: Data Validation

Dataset được kiểm tra trong quá trình đọc:

* Missing values
* Giá trị feature có thể chuyển sang số
* Cột `quality` có thể chuyển sang nhãn số
* Số lượng feature và target

Dataset hiện tại có 1.500 dòng, 6 feature và không có missing values. Nếu dữ liệu không hợp lệ hoặc không thể chuyển sang số, quá trình đọc dataset sẽ báo lỗi.

---

## Bước 4: Feature Engineering

Các feature được sử dụng trực tiếp để huấn luyện:

```text
weight
size
moisture
sugar_level
firmness
color_score
```

Do các feature có thang đo khác nhau nên sử dụng:

```text
StandardScaler
```

để chuẩn hóa dữ liệu trước khi đưa vào SVM.

---

# 6. Train/Test Split

Dataset được chia thành:

```text
80% → Training
20% → Testing
```

Sử dụng stratified split để giữ tỷ lệ các lớp.

---

# 7. SVM Model

Mô hình sử dụng:

```python
SVC
```

Pipeline:

```text
Input Features
      ↓
StandardScaler
      ↓
SVC
      ↓
Quality Prediction
```

Các kernel được thử nghiệm:

```text
rbf
linear
```

Các tham số được tuning:

```text
C
gamma
```

Sử dụng:

```python
GridSearchCV
```

để tìm cấu hình tốt nhất.

---

# 8. Train Model

Chạy:

```bash
py scripts\train_model.py
```

Sau khi train, model được lưu:

```text
artifacts/svm_model.joblib
```

Metrics được lưu:

```text
artifacts/metrics.json
```

---

# 9. Kết quả mô hình

Kết quả thử nghiệm hiện tại:

```text
Accuracy : 99.33%
F1 Macro : 99.33%
```

Cấu hình tốt nhất:

```text
Kernel : RBF
C      : 1.0
Gamma  : scale
```

Kết quả có thể thay đổi tùy vào cách chia dữ liệu hoặc quá trình training.

---

# 10. Prediction API

Đảm bảo FastAPI đang chạy ở `http://127.0.0.1:8000` trước khi chạy demo. Nếu chưa chạy, mở một terminal riêng:

```bash
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Sau đó mở Swagger tại `http://127.0.0.1:8000/docs` và gọi endpoint `/predict`.

Ví dụ input:

```text
weight      = 212.07
size        = 8.51
moisture    = 92.49
sugar_level = 13.84
firmness    = 8.24
color_score = 9.16
```

Thứ tự feature:

```text
weight
size
moisture
sugar_level
firmness
color_score
```

---

# 11. FastAPI

API được xây dựng bằng FastAPI.

Chạy server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API chạy tại:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 12. API Endpoints

## Train Dataset

```http
POST /train-dataset
```

Dùng để thực hiện quá trình training dataset.

Endpoint này đọc dataset mặc định tại `data/agricultural_product_quality.csv`, đánh giá model trên tập test và lưu model cùng metrics vào thư mục `artifacts/`.

## Train dữ liệu tự cung cấp

```http
POST /train
```

Ví dụ request:

```json
{
   "features": [[0, 0], [0, 1], [1, 0], [1, 1]],
   "labels": ["low", "high", "high", "low"]
}
```

Số dòng trong `features` và `labels` phải bằng nhau. Các dòng feature cũng phải có cùng số lượng giá trị.

## Health check

```http
GET /health
```

Ví dụ response:

```json
{
   "status": true,
   "model_ready": true
}
```

---

## Get Metrics

```http
GET /metrics
```

Trả về kết quả đánh giá model.

Ví dụ:

```json
{
   "samples": 1500,
   "train_samples": 1200,
   "test_samples": 300,
  "accuracy": 0.9933,
   "precision_macro": 0.9933,
   "recall_macro": 0.9933,
   "f1_macro": 0.9933,
   "best_params": {
      "classifier__C": 1.0,
      "classifier__gamma": "scale",
      "classifier__kernel": "rbf"
   }
}
```

---

## Prediction

```http
POST /predict
```

Input:

```json
{
  "features": [
    [
      212.07,
      8.51,
      92.49,
      13.84,
      8.24,
      9.16
    ]
  ]
}
```

Response:

```json
{
  "predictions": [2],
   "quality_labels": ["Cao"],
   "probabilities": [[...]],
   "confidence": [...],
   "conclusions": ["Nông sản được phân loại: Cao"]
}
```

Thứ tự 6 feature trong mỗi dòng bắt buộc là:

```text
weight, size, moisture, sugar_level, firmness, color_score
```

---

# 13. Test API bằng cURL

Sau khi chạy FastAPI:

```bash
curl.exe -X POST http://127.0.0.1:8000/predict ^
-H "Content-Type: application/json" ^
-d "{\"features\":[[212.07,8.51,92.49,13.84,8.24,9.16]]}"
```

---

# 14. Docker

Project sử dụng Docker để đóng gói FastAPI và SVM model.

## Build Docker Image

Trước tiên train model:

```bash
py scripts\train_model.py
```

Sau đó build image:

```bash
docker build -t local-svm-api .
```

Kiểm tra image:

```bash
docker images
```

---

## 15. Chạy Docker Container

Chạy:

```bash
docker run --rm -p 8000:8000 local-svm-api
```

Sau khi chạy thành công:

```text
Docker Container
      ↓
FastAPI
      ↓
Port 8000
```

Có thể truy cập:

```text
http://127.0.0.1:8000/docs
```

---

# 16. Docker + Ngrok

Để API có thể truy cập từ Internet, sử dụng Ngrok.

Kiến trúc:

```text
              Internet
                  │
                  ▼
          ┌──────────────┐
          │    Ngrok     │
          │ HTTPS Tunnel │
          └──────┬───────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Docker Container│
        │                 │
        │ FastAPI :8000   │
        │       ↓         │
        │    SVM Model    │
        └─────────────────┘
```

### Terminal 1 — Chạy Docker

```bash
docker run --rm -p 8000:8000 local-svm-api
```

### Terminal 2 — Chạy Ngrok

```bash
ngrok http 8000
```

Ngrok sẽ cung cấp một URL dạng:

```text
https://xxxx-xxxx.ngrok-free.app
```

---

# 17. Truy cập API qua Ngrok

Swagger:

```text
https://xxxx-xxxx.ngrok-free.app/docs
```

Prediction API:

```text
https://xxxx-xxxx.ngrok-free.app/predict
```

Ví dụ:

```bash
curl.exe -X POST https://xxxx-xxxx.ngrok-free.app/predict ^
-H "Content-Type: application/json" ^
-d "{\"features\":[[212.07,8.51,92.49,13.84,8.24,9.16]]}"
```

---

# 18. Sử dụng API từ Mobile App

Nếu sử dụng Expo Go hoặc ứng dụng mobile khác, không sử dụng:

```text
http://127.0.0.1:8000
```

Thay vào đó sử dụng URL Ngrok:

```text
https://xxxx-xxxx.ngrok-free.app
```

Ví dụ:

```javascript
const API_URL = "https://xxxx-xxxx.ngrok-free.app";
```

Gọi API:

```javascript
const response = await fetch(`${API_URL}/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    features: [
      [212.07, 8.51, 92.49, 13.84, 8.24, 9.16]
    ],
  }),
});

const data = await response.json();
```

---

# 19. Lưu ý về Ngrok

Ngrok chỉ đóng vai trò tạo đường hầm từ Internet vào API đang chạy trên máy local.

```text
Internet
   ↓
Ngrok
   ↓
localhost:8000
   ↓
Docker
   ↓
FastAPI
```

**Không cần cài Ngrok bên trong Docker.**

Chỉ cần:

```text
Terminal 1 → Docker
Terminal 2 → Ngrok
```

URL Ngrok miễn phí có thể thay đổi khi tạo tunnel mới.

---

# 20. Kiểm thử

Chạy:

```bash
py -m pytest -q
```

Mục đích kiểm tra:

* Load dataset
* Model
* Prediction
* API
* Các chức năng chính của project

---

# 21. Requirements

Cài đặt thư viện:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Các thư viện được khai báo trong `requirements.txt` gồm NumPy, scikit-learn, Joblib, FastAPI, Uvicorn, HTTPX và Pytest.

---

# 22. Quick Start

Nếu muốn chạy toàn bộ project từ đầu:

### 1. Clone project

```bash
git clone https://github.com/PhongNguyen2403/Mechine-Learning.git
```

### 2. Di chuyển vào project

```bash
cd Mechine-Learning
```

### 3. Cài thư viện

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

### 4. EDA

```bash
py scripts\eda.py
```

### 5. Train model

```bash
py scripts\train_model.py
```

### 6. Khởi động API

Mở terminal riêng và chạy:

```bash
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 7. Test model

Mở Swagger tại `http://127.0.0.1:8000/docs`, sau đó gọi `/train-dataset` để train
bằng dataset có sẵn và gọi `/predict` để kiểm tra dự đoán.

### 8. Build Docker

```bash
docker build -t local-svm-api .
```

### 9. Chạy Docker

```bash
docker run --rm -p 8000:8000 local-svm-api
```

### 10. Mở terminal mới và chạy Ngrok

```bash
ngrok http 8000
```

### 11. Sử dụng URL Ngrok

```text
https://xxxx-xxxx.ngrok-free.app
```

Swagger:

```text
https://xxxx-xxxx.ngrok-free.app/docs
```

---

# 23. Tổng quan hệ thống

```text
                     ┌──────────────────────┐
                     │ Agricultural Dataset │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ EDA / Data Validation│
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Feature Engineering  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ StandardScaler + SVM │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Trained Model      │
                     │ svm_model.joblib     │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │      FastAPI         │
                     │       :8000          │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │       Docker         │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │       Ngrok          │
                     │    HTTPS Tunnel      │
                     └──────────┬───────────┘
                                │
                                ▼
                    Internet / Mobile App
```

---

# 24. Mục tiêu của project

Project hướng tới việc minh họa quy trình triển khai một mô hình Machine Learning từ dữ liệu đến ứng dụng thực tế:

```text
Machine Learning
       +
FastAPI
       +
Docker
       +
Ngrok
       ↓
Public AI API
```

Hệ thống có thể được mở rộng để tích hợp với:

* Web Application
* Mobile Application
* Dashboard
* Các hệ thống quản lý khác
* Các mô hình Machine Learning khác
