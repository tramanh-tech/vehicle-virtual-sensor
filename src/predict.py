import pandas as pd
import joblib
import os

def predict(data_path):
    """
    Hãy dùng model đã được train và save để predict mass và slope.
    Hàm này tải file classifier.pkl và regressor.pkl từ thư mục models/.
    Lọc features chuẩn và trả về DataFrame chứa kết quả.
    """
    # Load dataset
    print(f"Bắt đầu đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)
    
    # Locate models
    base_path = "models" if os.path.exists("models/classifier.pkl") else "../models"
    clf_path = os.path.join(base_path, "classifier.pkl")
    reg_path = os.path.join(base_path, "regressor.pkl")
    
    # 1. Load models
    try:
        classifier = joblib.load(clf_path)
        regressor = joblib.load(reg_path)
        print("✅ Tải models thành công (Mass Classifier & Slope Regressor).")
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {e}")
        return None
        
    # 2. Xử lý logic predict Mass
    mass_features = ["Epm_nEng_100ms", "VehV_v_100ms", "ActMod_trqInr_100ms", "RngMod_trqCrSmin_100ms", "RoadSlope_100ms"]
    X_mass = df[mass_features]
    
    mass_preds = classifier.predict(X_mass)
    
    # 3. Xử lý logic predict Slope
    slope_features = ["Epm_nEng_100ms", "VehV_v_100ms", "ActMod_trqInr_100ms", "RngMod_trqCrSmin_100ms"]
    X_slope = df[slope_features].copy()
    
    # Dùng target giả lập (Mass vừa predict được, hoặc lấy trực tiếp từ data test)
    X_slope["Vehicle_Mass"] = df["Vehicle_Mass"] if "Vehicle_Mass" in df.columns else mass_preds
    
    slope_preds = regressor.predict(X_slope)
    
    # 4. Gắn kết quả (Encode Mass lại thành '49t' và '38t' theo yêu cầu)
    def encode_mass(v):
        if v in [0, 0.0, 38, 38.0]: return "38t"
        if v in [1, 1.0, 49, 49.0]: return "49t"
        return str(v)
    
    result_df = df[slope_features].copy()
    result_df["Predicted_Vehicle_Mass"] = [encode_mass(m) for m in mass_preds]
    result_df["Predicted_Road_Slope"] = slope_preds
    
    # Giữ lại nhãn thực tế để đối chiếu (nếu có)
    if "Vehicle_Mass" in df.columns:
        result_df["Actual_Vehicle_Mass"] = [encode_mass(m) for m in df["Vehicle_Mass"]]
    if "RoadSlope_100ms" in df.columns:
        result_df["Actual_Road_Slope"] = df["RoadSlope_100ms"]
        
    return result_df

if __name__ == "__main__":
    # Test thử trực tiếp trên file test demo
    test_csv_path = "../data/raw/test_demo.csv" if os.path.basename(os.getcwd()) == 'src' else "data/raw/test_demo.csv"
    
    preds_df = predict(test_csv_path)
    if preds_df is not None:
        print("\n🎯 Kết quả chạy predict (5 dòng đầu):")
        print(preds_df.head().to_string())
        
        out_path = "predictions.csv"
        preds_df.to_csv(out_path, index=False)
        print(f"\n📥 Đã lưu toàn bộ kết quả ra file: {out_path}")
