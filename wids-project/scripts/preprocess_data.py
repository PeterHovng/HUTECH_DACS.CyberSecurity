import pandas as pd

def preprocess_data(input_csv, output_csv):
    try:
        data = pd.read_csv(input_csv)
        # Xử lý NaN
        data.fillna(0, inplace=True)
        # Chuyển bool thành int
        data = data.astype({col: 'float' for col in data.select_dtypes(include=['bool']).columns})
        data.to_csv(output_csv, index=False)
        print(f"Dữ liệu đã được xử lý và lưu vào {output_csv}")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    preprocess_data('../dataset/processed_cse_cic_ids2018_malicious_benign.csv',
                    '../dataset/processed_cse_cic_ids2018_cleaned.csv')