Kiến trúc hệ thống WIDS
Tổng quan
Hệ thống Web Intrusion Detection System (WIDS) giám sát request HTTP/HTTPS, dự đoán tấn công bằng mô hình RandomForest, lưu log vào SQLite, và hiển thị trên dashboard React.js.
Thành phần

Giám sát request:
Công cụ: mitmproxy.
Chức năng: Bắt request, trích xuất đặc trưng, gửi tới API /predict.


Backend (FastAPI):
Endpoint: /, /predict, /logs.
Tích hợp: Mô hình RandomForest, SQLite.


Database (SQLite):
Lưu: Thời gian, IP, đặc trưng, nhãn, xác suất.


Frontend (React.js):
Hiển thị: Log tấn công, cảnh báo request độc hại.



Luồng dữ liệu

Request HTTP/HTTPS -> mitmproxy trích xuất đặc trưng.
Đặc trưng -> API /predict -> Dự đoán -> Lưu vào SQLite.
Dashboard -> Gọi API /logs -> Hiển thị log.

Công nghệ

Backend: FastAPI, SQLite, Scikit-learn.
Frontend: React.js, Tailwind CSS.
Giám sát: mitmproxy.

