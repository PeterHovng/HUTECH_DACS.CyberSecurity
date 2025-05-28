Web Intrusion Detection System (WIDS)
Hệ thống phát hiện tấn công web sử dụng Machine Learning (RandomForest) để phân tích request HTTP/HTTPS, lưu log tấn công vào SQLite, và hiển thị trên dashboard React.js.
Cài đặt

Yêu cầu:

Python 3.8+
Node.js (tùy chọn, nếu chạy React cục bộ)
mitmproxy (cho giám sát request)


Cài đặt thư viện:
cd backend
pip install -r requirements.txt


Chuẩn bị mô hình:

Đặt random-forest-classifier_model.pkl và rf_scaler.pkl vào thư mục trained_models/.


Chạy backend:
python backend/app.py


Chạy giám sát request (tùy chọn):
mitmdump -s backend/capture_requests.py


Chạy frontend:
cd frontend
python -m http.server 8001

Truy cập http://localhost:8001.


Endpoint API

GET /: Trả về thông điệp chào mừng.
POST /predict: Dự đoán nhãn (Benign/Malicious) và xác suất.
GET /logs: Trả về danh sách log tấn công.

Cấu trúc project

backend/: FastAPI, mitmproxy script, SQLite database.
frontend/: React.js dashboard.
trained_models/: Mô hình ML và scaler.
dataset/: Dữ liệu đầu vào.
scripts/: Script tiền xử lý.
docs/: Tài liệu.

Liên hệ

Nếu gặp vấn đề, liên hệ [your_email@example.com].