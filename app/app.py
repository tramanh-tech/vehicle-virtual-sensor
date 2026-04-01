import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Vehicle Virtual Sensor", layout="wide")

st.title("🚜 Vehicle Virtual Sensor")
st.markdown("Upload dữ liệu sensor (tệp `.csv`) để dự đoán các thông số (như `Vehicle_Mass` / `Road_Slope`...).")

# Upload file
uploaded_file = st.file_uploader("Chọn tệp CSV dữ liệu", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Dữ liệu đầu vào (Xem trước 5 dòng đầu):")
        st.dataframe(df.head())

        if st.button("🚀 Bắt đầu dự đoán"):
            model_path = None
            if os.path.exists("models/classifier.pkl"):
                model_path = "models/classifier.pkl"
            elif os.path.exists("../models/classifier.pkl"):
                model_path = "../models/classifier.pkl"
            
            if model_path is None:
                st.warning("⚠️ Không tìm thấy file model `classifier.pkl` trong thư mục `models/`.")
            else:
                with st.spinner("Đang tải mô hình và thực hiện dự đoán..."):
                    try:
                        # Load models
                        classifier = joblib.load(model_path)
                        reg_path = model_path.replace("classifier.pkl", "regressor.pkl")
                        regressor = joblib.load(reg_path) if os.path.exists(reg_path) else None
                        
                        # Data Validation & Filtering
                        # Các features để dự đoán Mass (bao gồm RoadSlope input)
                        mass_features = [
                            "Epm_nEng_100ms", "VehV_v_100ms", 
                            "ActMod_trqInr_100ms", "RngMod_trqCrSmin_100ms", 
                            "RoadSlope_100ms"
                        ]
                        
                        # Kiểm tra xem đủ 5 cột cần thiết không (Giả định RoadSlope có sẵn từ csv test)
                        missing_cols = [col for col in mass_features if col not in df.columns]
                        if missing_cols:
                            st.error(f"❌ File CSV của bạn bị thiếu các cột bắt buộc sau để chạy mô hình Mass: {missing_cols}")
                        else:
                            # 1. Dự đoán Mass
                            X_mass = df[mass_features]
                            mass_preds = classifier.predict(X_mass)
                            
                            # Lọc lại Dataframe để khởi tạo bảng Kết quả (hiển thị 4 cột sensor chung)
                            sensor_cols = ["Epm_nEng_100ms", "VehV_v_100ms", "ActMod_trqInr_100ms", "RngMod_trqCrSmin_100ms"]
                            result_df = df[sensor_cols].copy()
                            
                            # Encode lại Prediction của Mass (ví dụ 49.0 thành "49t")
                            def encode_mass(val):
                                if val in [0, 0.0, 38, 38.0]: return "38t"
                                if val in [1, 1.0, 49, 49.0]: return "49t"
                                return str(val)
                            
                            result_df["Predicted_Vehicle_Mass"] = [encode_mass(m) for m in mass_preds]
                            
                            if "Vehicle_Mass" in df.columns:
                                result_df["Actual_Vehicle_Mass"] = [encode_mass(m) for m in df["Vehicle_Mass"]]
                                
                            # 2. Dự đoán Slope (Nếu có regressor)
                            if regressor is not None:
                                # Feature để đoán Slope bao gồm Mass (lấy Actual nếu có trong CSV, hoặc lấy Predicted nếu không có)
                                slope_features = sensor_cols.copy()
                                slope_features.append("Vehicle_Mass")
                                
                                X_slope = df[sensor_cols].copy()
                                # Dùng Mass thực tế nếu test_demo có, nếu app thật thì dùng mảng mass_preds
                                X_slope["Vehicle_Mass"] = df["Vehicle_Mass"] if "Vehicle_Mass" in df.columns else mass_preds
                                
                                slope_preds = regressor.predict(X_slope)
                                result_df["Predicted_Road_Slope"] = slope_preds
                                
                                if "RoadSlope_100ms" in df.columns:
                                    # RoadSlope ban đầu được dùng để predict Mass, nay thành Actual_Slope để đối chiếu
                                    result_df["Actual_Road_Slope"] = df["RoadSlope_100ms"]
                                    
                            st.success("✅ Dự đoán thành công (đã bao gồm biến đổi nhãn Mass và dự đoán Slope)!")
                            st.write("### 🎯 Kết quả dự đoán (Rút gọn):")
                            st.dataframe(result_df)
                            
                            csv_data = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Tải tệp kết quả (CSV)", data=csv_data, file_name="predictions.csv", mime="text/csv")
                            
                    except Exception as e:
                        st.error(f"❌ Có lỗi xảy ra trong quá trình dự đoán: {e}")
                        
    except Exception as e:
         st.error(f"❌ Lỗi khi đọc file CSV: {e}")
