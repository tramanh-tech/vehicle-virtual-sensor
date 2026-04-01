import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # TODO: Cập nhật hàm xử lý rác/chuẩn hoá dữ liệu (missing values, scale, etc) sau.
    pass

def create_demo_test_set(train_path, test_path, n_samples=50):
    """
    Hàm tạo nhanh một file CSV nhỏ (lấy 50 dòng từ train) để upload test Demo App
    """
    if not os.path.exists(train_path):
        print(f"❌ Lỗi: Không tìm thấy file gốc tại {train_path}")
        return
        
    print(f"⏳ Đọc dữ liệu từ {train_path}...")
    df = pd.read_csv(train_path)
    
    # Lấy 50 dòng đầu tiên (hoặc lấy df.sample(n=50) nếu bạn muốn xáo ngẫu nhiên)
    df_demo = df.head(n_samples) 
    
    df_demo.to_csv(test_path, index=False)
    print(f"✅ Đã tạo thành công file test chứa {len(df_demo)} dòng tại: {test_path}")

if __name__ == "__main__":
    # Kiểm tra đường dẫn khi file này được chạy tuỳ thuộc vào thư mục gốc của Terminal
    current_dir = os.path.basename(os.getcwd())
    
    if current_dir == "src":
        train_file = "../data/raw/train.csv"
        demo_file = "../data/raw/test_demo.csv"
    else:
        train_file = "data/raw/train.csv"
        demo_file = "data/raw/test_demo.csv"
        
    create_demo_test_set(train_file, demo_file, n_samples=50)
