import sqlite3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import csv
import json
import io  # Thêm import io

app = FastAPI(title="WIDS API")

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải mô hình và scaler
try:
#   model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\random_forest_multi_class_model.pkl')
    model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\decision_tree_multi_class_model.pkl')
#   model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\xgboost_multi_class_model.pkl')
#   model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\ensemble_dt_xg_model.pkl')
    scaler = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\rf_scaler_multi_class.pkl')
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Error loading model/scaler: {str(e)}")

# Đọc dataset để lấy đặc trưng
try:
    data = pd.read_csv(r'C:\Users\garan\OneDrive\Máy tính\wids-project\dataset\processed_cse_cic_ids2018_multi_class_dataset.csv')
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

# Khởi tạo database SQLite
def init_db():
    conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attack_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  ip_address TEXT,
                  features TEXT,
                  prediction TEXT,
                  probabilities TEXT,
                  attack_type TEXT DEFAULT 'N/A')''')
    conn.commit()
    conn.close()

init_db()

# Định nghĩa cấu trúc dữ liệu
class InputData(BaseModel):
    features: List[float]
    ip_address: Optional[str] = "create ip address"

class PredictionResponse(BaseModel):
    prediction: str
    probability: dict
    attack_type: str

@app.post("/predict", tags=["WIDS Operations"], response_model=PredictionResponse)
async def predict(data: InputData):
    try:
        # Chuyển đổi features thành numpy array
        features = np.array(data.features).reshape(1, -1)

        # Kiểm tra số lượng đặc trưng
        expected_features = scaler.n_features_in_
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Số lượng đặc trưng không đúng. Mong đợi {expected_features}, nhận được {features.shape[1]}."
            )

        # Chuẩn hóa dữ liệu
        features_scaled = scaler.transform(features)

        # Dự đoán
        probabilities = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        prob_dict = dict(zip(model.classes_, probabilities))

        # Xác định attack_type
        attack_type = prediction if prediction != "Benign" else "N/A"

        # Lưu vào database
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO attack_logs (timestamp, ip_address, features, prediction, probabilities, attack_type) VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, data.ip_address, str(data.features), prediction, str(prob_dict), attack_type))
        conn.commit()
        conn.close()

        return {
            "prediction": prediction,
            "probability": prob_dict,
            "attack_type": attack_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình dự đoán: {str(e)}")

@app.get("/random_features", tags=["WIDS Operations"])
async def random_features():
    try:
        row_index = np.random.randint(0, len(data))
        new_data_row = data.iloc[row_index]

        # Loại bỏ cột Label và chuyển thành danh sách
        features = new_data_row.drop('Label').tolist()
        features = [0.0 if pd.isna(x) else float(x) for x in features]

        return str(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo đặc trưng ngẫu nhiên: {str(e)}")

@app.get("/get_features", tags=["WIDS Operations"])
async def get_features(row_index: int):
    try:
        # Kiểm tra row_index hợp lệ
        if row_index < 0 or row_index >= len(data):
            raise HTTPException(
                status_code=400,
                detail=f"row_index phải nằm trong khoảng [0, {len(data)-1}]"
            )

        # Lấy dòng dữ liệu từ dataset
        new_data_row = data.iloc[row_index]

        # Loại bỏ cột Label và chuyển thành danh sách
        features = new_data_row.drop('Label').tolist()
        features = [0.0 if pd.isna(x) else float(x) for x in features]

        # Trả về chuỗi dạng [x, y, z, ...]
        return str(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy đặc trưng: {str(e)}")

@app.get("/logs", tags=["WIDS Operations"])
async def get_logs():
    try:
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
        c = conn.cursor()
        c.execute("SELECT * FROM attack_logs ORDER BY timestamp DESC")
        logs = c.fetchall()
        conn.close()
        return [{
            "id": log[0],
            "timestamp": log[1],
            "ip_address": log[2],
            "features": log[3],
            "prediction": log[4],
            "probabilities": eval(log[5]),
            "attack_type": log[6] if log[6] else "N/A"
        } for log in logs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy log: {str(e)}")

@app.get("/clear_logs", tags=["WIDS Operations"])
async def clear_logs():
    try:
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
        c = conn.cursor()
        c.execute("DELETE FROM attack_logs")
        conn.commit()
        conn.close()
        return {"message": "All logs have been cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa logs: {str(e)}")

@app.get("/search_logs", tags=["WIDS Operations"])
async def search_logs(
    ip_address: Optional[str] = Query(None, description="Search by IP address"),
    prediction: Optional[str] = Query(None, description="Search by prediction"),
    attack_type: Optional[str] = Query(None, description="Search by attack type"),
    start_time: Optional[str] = Query(None, description="Start time (YYYY-MM-DD HH:MM:SS)"),
    end_time: Optional[str] = Query(None, description="End time (YYYY-MM-DD HH:MM:SS)")
):
    try:
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
        c = conn.cursor()

        # Xây dựng câu truy vấn động
        query = "SELECT * FROM attack_logs WHERE 1=1"
        params = []
        
        if ip_address:
            query += " AND ip_address LIKE ?"
            params.append(f"%{ip_address}%")
        if prediction:
            query += " AND prediction LIKE ?"
            params.append(f"%{prediction}%")
        if attack_type:
            query += " AND attack_type LIKE ?"
            params.append(f"%{attack_type}%")
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC"
        c.execute(query, params)
        logs = c.fetchall()
        conn.close()

        return [{
            "id": log[0],
            "timestamp": log[1],
            "ip_address": log[2],
            "features": log[3],
            "prediction": log[4],
            "probabilities": eval(log[5]),
            "attack_type": log[6] if log[6] else "N/A"
        } for log in logs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm log: {str(e)}")

@app.get("/export_logs", tags=["WIDS Operations"])
async def export_logs(format: str = Query("csv", description="Export format (csv or json)")):
    try:
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
        c = conn.cursor()
        c.execute("SELECT * FROM attack_logs ORDER BY timestamp DESC")
        logs = c.fetchall()
        conn.close()

        # Kiểm tra nếu không có log
        if not logs:
            raise HTTPException(status_code=404, detail="No logs available to export.")

        logs_data = []
        for log in logs:
            try:
                probabilities = eval(log[5])  # Chuyển đổi probabilities từ string sang dict
            except Exception as e:
                probabilities = {}  # Nếu lỗi, trả về dict rỗng
            logs_data.append({
                "id": log[0],
                "timestamp": log[1],
                "ip_address": log[2],
                "features": log[3],
                "prediction": log[4],
                "probabilities": probabilities,
                "attack_type": log[6] if log[6] else "N/A"
            })

        if format.lower() == "csv":
            # Tạo nội dung CSV
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=logs_data[0].keys())
            writer.writeheader()
            writer.writerows(logs_data)
            return Response(
                content=output.getvalue().encode(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=wids_logs_{datetime.now().strftime('%Y-%m-%d')}.csv"}
            )
        elif format.lower() == "json":
            # Tạo nội dung JSON
            json_data = json.dumps(logs_data, default=str)
            return Response(
                content=json_data.encode(),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=wids_logs_{datetime.now().strftime('%Y-%m-%d')}.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xuất log: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)