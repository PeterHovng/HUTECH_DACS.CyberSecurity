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
import io
import re

app = FastAPI(title="WIDS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
#   model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\random_forest_multi_class_model.pkl')
    model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\decision_tree_multi_class_model.pkl')
#   model = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\xgboost_multi_class_model.pkl')
    scaler = joblib.load(r'C:\Users\garan\OneDrive\Máy tính\wids-project\trained_models\rf_scaler_multi_class.pkl')
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Error loading model/scaler: {str(e)}")

try:
    data = pd.read_csv(r'C:\Users\garan\OneDrive\Máy tính\wids-project\dataset\processed_cse_cic_ids2018_multi_class_dataset.csv')
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

def init_attack_db():
    conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attack_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  ip_address TEXT,
                  features TEXT,
                  prediction TEXT,
                  probabilities TEXT,
                  attack_type TEXT DEFAULT 'N/A',
                  auto_blocked INTEGER DEFAULT 0)''')
    c.execute("PRAGMA table_info(attack_logs)")
    columns = [info[1] for info in c.fetchall()]
    if 'auto_blocked' not in columns:
        c.execute("ALTER TABLE attack_logs ADD COLUMN auto_blocked INTEGER DEFAULT 0")
    conn.commit()
    conn.close()

def init_ipaddress_db():
    conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\ipaddress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS blocked_ips
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ip_address TEXT UNIQUE,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_attack_db()
init_ipaddress_db()

class InputData(BaseModel):
    features: List[float]
    ip_address: Optional[str] = "create ip address"

class PredictionResponse(BaseModel):
    prediction: str
    probability: dict
    attack_type: str
    auto_blocked: bool

class BlockIPRequest(BaseModel):
    ip_address: str

def is_valid_ip(ip):
    pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    if not re.match(pattern, ip):
        return False
    return all(0 <= int(part) <= 255 for part in ip.split('.'))

def block_ip_internal(ip_address):
    if not is_valid_ip(ip_address):
        return False

    conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\ipaddress.db')
    c = conn.cursor()
    c.execute("SELECT ip_address FROM blocked_ips WHERE ip_address = ?", (ip_address,))
    if c.fetchone():
        conn.close()
        return False

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO blocked_ips (ip_address, timestamp) VALUES (?, ?)", (ip_address, timestamp))
    conn.commit()
    conn.close()
    return True

@app.post("/predict", tags=["WIDS Operations"], response_model=PredictionResponse)
async def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        expected_features = scaler.n_features_in_
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Số lượng đặc trưng không đúng. Mong đợi {expected_features}, nhận được {features.shape[1]}."
            )

        features_scaled = scaler.transform(features)
        probabilities = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        attack_type = prediction if prediction != "Benign" else "N/A"

        confidence_score = max(probabilities) * 100
        auto_blocked = False
        if prediction != "Benign" and confidence_score > 70 and data.ip_address != "create ip address":
            auto_blocked = block_ip_internal(data.ip_address)

        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\attacks.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO attack_logs (timestamp, ip_address, features, prediction, probabilities, attack_type, auto_blocked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, data.ip_address, str(data.features), prediction, str(prob_dict), attack_type, int(auto_blocked)))
        conn.commit()
        conn.close()

        return {
            "prediction": prediction,
            "probability": prob_dict,
            "attack_type": attack_type,
            "auto_blocked": auto_blocked
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình dự đoán: {str(e)}")

@app.post("/block_ip", tags=["WIDS Operations"])
async def block_ip(request: BlockIPRequest):
    try:
        ip_address = request.ip_address
        if not is_valid_ip(ip_address):
            raise HTTPException(status_code=400, detail="Định dạng IP không hợp lệ. Vui lòng nhập đúng định dạng (xxx.xxx.xxx.xxx).")

        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\ipaddress.db')
        c = conn.cursor()
        c.execute("SELECT ip_address FROM blocked_ips WHERE ip_address = ?", (ip_address,))
        if c.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail=f"IP {ip_address} đã được chặn trước đó.")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO blocked_ips (ip_address, timestamp) VALUES (?, ?)", (ip_address, timestamp))
        conn.commit()
        conn.close()
        return {"message": f"IP {ip_address} đã được chặn thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chặn IP: {str(e)}")

@app.post("/unblock_ip", tags=["WIDS Operations"])
async def unblock_ip(request: BlockIPRequest):
    try:
        ip_address = request.ip_address
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\ipaddress.db')
        c = conn.cursor()
        c.execute("SELECT ip_address FROM blocked_ips WHERE ip_address = ?", (ip_address,))
        if not c.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail=f"IP {ip_address} không có trong danh sách chặn.")

        c.execute("DELETE FROM blocked_ips WHERE ip_address = ?", (ip_address,))
        conn.commit()
        conn.close()
        return {"message": f"IP {ip_address} đã được bỏ chặn thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi bỏ chặn IP: {str(e)}")

@app.get("/blocked_ips", tags=["WIDS Operations"])
async def get_blocked_ips():
    try:
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\ipaddress.db')
        c = conn.cursor()
        c.execute("SELECT ip_address, timestamp FROM blocked_ips ORDER BY timestamp DESC")
        blocked_ips = c.fetchall()
        conn.close()
        return [{"ip_address": ip[0], "timestamp": ip[1]} for ip in blocked_ips]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách IP bị chặn: {str(e)}")

@app.get("/export_blocked_ips", tags=["WIDS Operations"])
async def export_blocked_ips(format: str = Query("csv", description="Export format (csv or json)")):
    try:
        conn = sqlite3.connect(r'C:\Users\garan\OneDrive\Máy tính\wids-project\backend\database\ipaddress.db')
        c = conn.cursor()
        c.execute("SELECT ip_address, timestamp FROM blocked_ips ORDER BY timestamp DESC")
        blocked_ips = c.fetchall()
        conn.close()
        if not blocked_ips:
            raise HTTPException(status_code=404, detail="No blocked IPs available to export.")
        
        blocked_ips_data = [{"ip_address": ip[0], "timestamp": ip[1]} for ip in blocked_ips]
        
        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["ip_address", "timestamp"])
            writer.writeheader()
            writer.writerows(blocked_ips_data)
            return Response(
                content=output.getvalue().encode(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=blocked_ips_{datetime.now().strftime('%Y-%m-%d')}.csv"}
            )
        elif format.lower() == "json":
            json_data = json.dumps(blocked_ips_data, default=str)
            return Response(
                content=json_data.encode(),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=blocked_ips_{datetime.now().strftime('%Y-%m-%d')}.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xuất danh sách IP bị chặn: {str(e)}")

@app.get("/random_features", tags=["WIDS Operations"])
async def random_features():
    try:
        row_index = np.random.randint(0, len(data))
        new_data_row = data.iloc[row_index]
        features = new_data_row.drop('Label').tolist()
        features = [0.0 if pd.isna(x) else float(x) for x in features]
        return str(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo đặc trưng ngẫu nhiên: {str(e)}")

@app.get("/get_features", tags=["WIDS Operations"])
async def get_features(row_index: int):
    try:
        if row_index < 0 or row_index >= len(data):
            raise HTTPException(
                status_code=400,
                detail=f"row_index phải nằm trong khoảng [0, {len(data)-1}]"
            )
        new_data_row = data.iloc[row_index]
        features = new_data_row.drop('Label').tolist()
        features = [0.0 if pd.isna(x) else float(x) for x in features]
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
            "attack_type": log[6] if log[6] else "N/A",
            "auto_blocked": bool(log[7])
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
            "attack_type": log[6] if log[6] else "N/A",
            "auto_blocked": bool(log[7])
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
        if not logs:
            raise HTTPException(status_code=404, detail="No logs available to export.")
        logs_data = []
        for log in logs:
            try:
                probabilities = eval(log[5])
            except Exception as e:
                probabilities = {}
            logs_data.append({
                "id": log[0],
                "timestamp": log[1],
                "ip_address": log[2],
                "features": log[3],
                "prediction": log[4],
                "probabilities": probabilities,
                "attack_type": log[6] if log[6] else "N/A",
                "auto_blocked": bool(log[7])
            })
        if format.lower() == "csv":
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