import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

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
    
    keep_cols = ['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date'] + v_cols
    df = df[keep_cols].copy()
    
    # Đổi tên cột SCATS Number thành SCATS_ID, Lat/Lon
    df.rename(columns={'SCATS Number': 'SCATS_ID', 'NB_LATITUDE': 'lat', 'NB_LONGITUDE': 'lon'}, inplace=True)

    # Loại "Null Island" (0.0, 0.0) vì thường là giá trị mặc định do lỗi cảm biến.
    # Nếu không loại, KNN adjacency và visualization sẽ bị scale sai khiến toàn bộ mạng bị "dồn chấm".
    coord_eps = 1e-6
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    is_null_island = (df['lat'].abs() <= coord_eps) & (df['lon'].abs() <= coord_eps)
    
    # Trích xuất điểm tọa độ trung bình (chỉ dùng các hàng có tọa độ hợp lệ).
    # Điều này chống được lỗi 1 trạm chứa tọa độ không nhất quán sinh ra trùng lặp dòng (shape = 4, 6)
    # và giảm ảnh hưởng từ các bản ghi tọa độ mặc định 0.0, 0.0.
    coords_df = df.loc[~is_null_island].groupby('SCATS_ID')[['lat', 'lon']].mean().reset_index()
    df = df.drop(columns=['lat', 'lon'])
    
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

    # Tổng lưu lượng của các hướng đi cùng SCATS_ID & Timestamp
    df_long = df_long.groupby(['SCATS_ID', 'Timestamp'])['Traffic_Volume'].sum().reset_index()
    
    # Gắn trả lại tọa độ lat, lon vào
    df_long = df_long.merge(coords_df, on='SCATS_ID', how='left')

    # Nếu một SCATS_ID chỉ tồn tại với tọa độ (0,0) thì lat/lon sẽ là NaN sau merge -> bỏ toàn bộ trạm đó.
    df_long = df_long.dropna(subset=['lat', 'lon']).copy()
    
    # Thứ trong tuần (0=Thứ Hai ... 6=Chủ nhật, pandas dayofweek)
    df_long['day_of_week'] = df_long['Timestamp'].dt.dayofweek
    df_long['hour_of_day'] = df_long['Timestamp'].dt.hour
    # Chu kỳ 24h (theo giờ 0–23)
    h = df_long['hour_of_day'].to_numpy(dtype=np.float64)
    df_long['hour_sin'] = np.sin(2 * np.pi * h / 24.0)
    df_long['hour_cos'] = np.cos(2 * np.pi * h / 24.0)

    # Chu kỳ trong ngày theo bước 15 phút (0–95 khung/ngày) — tương đương “1 ngày” chi tiết hơn 1h
    slot_96 = (df_long['Timestamp'].dt.hour * 4 + df_long['Timestamp'].dt.minute // 15).to_numpy(dtype=np.float64)
    df_long['slot_sin'] = np.sin(2 * np.pi * slot_96 / 96.0)
    df_long['slot_cos'] = np.cos(2 * np.pi * slot_96 / 96.0)

    # Chu kỳ thứ trong tuần (T2–CN): 7 bước
    dow = df_long['day_of_week'].to_numpy(dtype=np.float64)
    df_long['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df_long['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    df_long['is_weekend'] = (df_long['day_of_week'] >= 5).astype(int)
    df_long['is_rush_hour'] = df_long['hour_of_day'].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    )
    df_long['is_night'] = df_long['hour_of_day'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)

    # Lag theo từng trạm (15 phút: lag 4 ≈ 1h, lag 96 ≈ 1 ngày)
    df_long.sort_values(['SCATS_ID', 'Timestamp'], inplace=True)
    g = df_long.groupby('SCATS_ID', sort=False)['Traffic_Volume']
    df_long['traffic_lag_1'] = g.shift(1)
    df_long['traffic_lag_4'] = g.shift(4)
    df_long['traffic_lag_96'] = g.shift(96)
    df_long[['traffic_lag_1', 'traffic_lag_4', 'traffic_lag_96']] = df_long[
        ['traffic_lag_1', 'traffic_lag_4', 'traffic_lag_96']
    ].fillna(0.0)

    df_long.sort_values(by=['Timestamp', 'SCATS_ID'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    return df_long

def build_adjacency_matrix(df_long, k=3):
    """
    Xây dựng ma trận kề dựa trên tọa độ không gian (vĩ độ, kinh độ).
    Lưu ý: Dữ liệu DF_long cần phải có cột 'lat', 'lon'.
    """
    nodes = sorted(df_long['SCATS_ID'].unique())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    coords = df_long.groupby('SCATS_ID')[['lat', 'lon']].mean().loc[nodes].values

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    N = len(nodes)
    A = np.zeros((N, N))

    for i in range(N):
        for j_idx in indices[i][1:]:
            dist = np.linalg.norm(coords[i] - coords[j_idx])
            A[i][j_idx] = 1 / (dist + 1e-6)
            A[j_idx][i] = A[i][j_idx]

    return A, node_to_idx

def build_tensor(df_long, node_to_idx, feature_cols):
    """
    Tạo cấu trúc Tensor Không - Thời gian: (T_Time, N_Node, F_Feature)
    """
    times = sorted(df_long['Timestamp'].unique())
    nodes = list(node_to_idx.keys())

    T = len(times)
    N = len(nodes)
    F = len(feature_cols)

    tensor = np.zeros((T, N, F))
    
    # Tối ưu thời gian duyệt bằng cách group / pivot nhanh
    df_indexed = df_long.set_index(['Timestamp', 'SCATS_ID'])[feature_cols]
    
    for t_idx, t in enumerate(times):
        try:
            # Lấy data tại mốc t cho toàn bộ các node
            df_t = df_indexed.loc[t]
            for node in df_t.index:
                n_idx = node_to_idx[node]
                tensor[t_idx, n_idx, :] = df_t.loc[node].values
        except KeyError:
            continue
            
    return tensor

def create_st_sequences(tensor, seq_len=96):
    """
    Tạo cửa sổ trượt tịnh tiến theo KHUNG THỜI GIAN trên TẤT CẢ các trạm.
    Đầu vào: tensor (T, N, F)
    Đầu ra: 
       X shape (mẫu, thời_điểm, trạm, features)
       y shape (mẫu, trạm) - lưu lượng xe của trạm (giả định ở cột 0) ở bước tiếp theo
    """
    X, y = [], []
    T = tensor.shape[0]
    
    for i in range(T - seq_len):
        X.append(tensor[i : i + seq_len])
        # Dự đoán lưu lượng (feature 0) cho CẢ KHỐI CÁC TRẠM
        y.append(tensor[i + seq_len, :, 0])  
        
    return np.array(X), np.array(y)


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
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/SCATS_data.csv')
    
    print("Loading data...")
    data = load_data(data_path)
        
    print(f"Shape of initial data (Wide Format): {data.shape}")
    
    # 1. Reshape dữ liệu
    print("\n1. Reshaping data...")
    df_long = reshape_data_sum(data)
    print("Ví dụ Dữ liệu Long Format nhận được:")
    print(df_long.head())
    print(f"Hình dạng dữ liệu sau biến đổi: {df_long.shape}")
    
    # sequence_length = 96 
    
    # Phục vụ để code cũ tránh báo lỗi (vì Test Main Script cũ bị xóa create_lstm_sequences)
    print("Vui lòng kích hoạt script train_gcn_lstm.py để kiểm tra module đồ thị (Graph Convolution)...")