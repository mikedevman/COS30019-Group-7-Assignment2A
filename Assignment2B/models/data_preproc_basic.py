import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# def load_xlsx_file(xlsx_file):
#     df = pd.read_excel(xlsx_file)
#     return df
# data = load_xlsx_file('data.xlsx')
# print(data.head())
# data.to_csv('data.csv', index=False)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def reshape_data_sum(df):
    """
    Chuyển đổi dữ liệu từ Wide Format sang Long Format.
    Bao gồm SCATS_ID, Timestamp (kết hợp Ngày và Giờ), và Traffic_Volume.
    """
    # Lấy các cột lưu lượng V00 đến V95
    v_cols = [f'V{i:02d}' for i in range(96)]
    v_cols = [col for col in v_cols if col in df.columns]
    
    keep_cols = ['SCATS Number', 'Date'] + v_cols
    df = df[keep_cols].copy()
    
    # Đổi tên cột SCATS Number thành SCATS_ID
    df.rename(columns={'SCATS Number': 'SCATS_ID'}, inplace=True)
    
    # Chuẩn hóa cột Date về 00:00:00 của ngày đó
    # Một số giá trị Date có sẵn giờ (ví dụ: '2006-10-01 00:15:00'), ta lấy phần ngày normalize
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    
    # Reshaping từ bảng ngang sang bảng dọc (Long Format) bằng phương thức melt
    df_long = df.melt(
        id_vars=['SCATS_ID', 'Date'],
        value_vars=v_cols,
        var_name='TimeInterval',
        value_name='Traffic_Volume'
    )
    
    # Xử lý thời gian từ chuỗi V00 -> V95 thành timedelta
    # Mỗi khoảng (interval) tương ứng với 15 phút
    minutes_offset = df_long['TimeInterval'].str.replace('V', '').astype(int) * 15
    df_long['TimeDelta'] = pd.to_timedelta(minutes_offset, unit='m')
    
    # Kết hợp Date và khoảng thời gian lệch để ra Timestamp chính xác
    df_long['Timestamp'] = df_long['Date'] + df_long['TimeDelta']
    
    # Bỏ các cột không còn cần thiết, giữ lại cấu trúc yêu cầu
    df_long = df_long[['SCATS_ID', 'Timestamp', 'Traffic_Volume']]

    #Tổng lưu lượng của các hướng đi cùng SCATS_ID & Timestamp
    df_long = df_long.groupby(['SCATS_ID', 'Timestamp'])['Traffic_Volume'].sum().reset_index()
    
    # Dữ liệu phải được nhóm theo SCATS_ID bằng cách sắp xếp theo trạm và thời gian
    df_long.sort_values(by=['SCATS_ID', 'Timestamp'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    #days of week
    df_long['day_of_week'] = df_long['Timestamp'].dt.dayofweek

    #hour of day
    df_long['hour_of_day'] = df_long['Timestamp'].dt.hour

    #is_weekend
    df_long['is_weekend'] = df_long['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # is_rush_hour: Giờ cao điểm sáng (6h - 9h) và chiều tối (16h - 19h)
    df_long['is_rush_hour'] = df_long['hour_of_day'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)

    #is_night
    df_long['is_night'] = df_long['hour_of_day'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
    
    return df_long

def create_lstm_sequences(df_long, seq_length=96, train_ratio=0.7, val_ratio=0.15):
    """
    Tạo dữ liệu sliding window cho mô hình LSTM/GRU.
    Đồng thời chia tập Train/Val/Test theo thời gian cho TỪNG trạm để mô hình
    không bị "mù" dữ liệu của các trạm nằm ở đoạn cuối.
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    # Chú ý: Lấy tất cả các features. Biến dự đoán (y) nằm ở cột đầu tiên (Traffic_Volume)
    feature_cols = ['Traffic_Volume', 'day_of_week', 'hour_of_day', 'is_weekend', 'is_rush_hour', 'is_night']
    
    # Tuyệt đối không tạo sliding window bắc cầu: 
    # lặp qua từng nhóm dữ liệu của từng cụm SCATS_ID
    for scats_id, group in df_long.groupby('SCATS_ID'):
        # Sort tiếp lại cho chắc chắn đúng trình tự thời gian
        group = group.sort_values('Timestamp')
        
        # Trích xuất toàn bộ ma trận features
        data_values = group[feature_cols].values
        
        n = len(data_values)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Tạo Sliding window với kích thước seq_length
        for i in range(n - seq_length):
            x_seq = data_values[i : i + seq_length]
            y_target = data_values[i + seq_length, 0] # Chỉ lấy cột 0 (Traffic_Volume) làm y_target
            target_idx = i + seq_length
            
            # Chia tập tuần tự cho từng trạm
            if target_idx < train_end:
                X_train.append(x_seq)
                y_train.append(y_target)
            elif target_idx < val_end:
                X_val.append(x_seq)
                y_val.append(y_target)
            else:
                X_test.append(x_seq)
                y_test.append(y_target)
            
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Dữ liệu tự nhiên đã là 3D (samples, seq_length, features) nên không cần expand_dims nữa
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def perform_eda(df_long):
    sns.set_theme(style="whitegrid")

    # 1. Vẽ chuỗi thời gian của 1 trạm trong 1 tuần đầu tiên
    plt.figure(figsize=(15, 5))
    sample_scats = df_long['SCATS_ID'].unique()[0] # Lấy trạm đầu tiên làm ví dụ
    sample_data = df_long[(df_long['SCATS_ID'] == sample_scats) & 
                          (df_long['Timestamp'] < df_long['Timestamp'].min() + pd.Timedelta(days=7))]
    
    plt.plot(sample_data['Timestamp'], sample_data['Traffic_Volume'], marker='.', linestyle='-')
    plt.title(f'Traffic Volume in 1 week at station {sample_scats}', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Traffic Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. So sánh mô hình giao thông Ngày thường vs Cuối tuần theo Giờ
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_long, x='hour_of_day', y='Traffic_Volume', hue='is_weekend', errorbar=None, marker='o')
    plt.title('Average Traffic Volume by Hour of Day (Weekdays vs Weekend)', fontsize=14)
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Traffic Volume')
    plt.legend(title='Day in Week', labels=['Weekdays (T2-T6)', 'Weekend (T7-CN)'])
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.show()

    # 3. Phân bố dữ liệu (Boxplot) để xem Outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_long['Traffic_Volume'])
    plt.title('Distribution of Total Traffic Volume in each 15 minutes', fontsize=14)
    plt.xlabel('Traffic Volume')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    file_path = '../data.csv'
    
    print("Loading data...")
    try:
        data = load_data('data.csv')
    except:
        data = load_data('../data.csv')
        
    print(f"Shape of initial data (Wide Format): {data.shape}")
    
    # 1. Reshape dữ liệu
    print("\n1. Reshaping data...")
    df_long = reshape_data_sum(data)
    print("Ví dụ Dữ liệu Long Format nhận được:")
    print(df_long.head())
    print(f"Hình dạng dữ liệu sau biến đổi: {df_long.shape}")
    
    # 2 & 3. Tạo dữ liệu sliding window (LSTM/GRU) theo từng nhóm và chia tập
    # Tham số cấu hình: seq_length = 96 (dữ liệu một ngày cho chuỗi 15 phút = 24 * 4)
    sequence_length = 96 
    print(f"\n2 & 3 & 4. Đang tạo dữ liệu Sliding Window và chia Train/Val/Test (seq_length/time steps = {sequence_length})...")
    X_train, y_train, X_val, y_val, X_test, y_test = create_lstm_sequences(df_long, seq_length=sequence_length, train_ratio=0.7, val_ratio=0.15)
    
    print("\nDữ liệu cung cấp cho LSTM/GRU đã hoàn tất!")
    print(f" - Tập train: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f" - Tập validation: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f" - Tập test: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # 5. EDA
    print("Bắt đầu vẽ biểu đồ EDA...")
    perform_eda(df_long)