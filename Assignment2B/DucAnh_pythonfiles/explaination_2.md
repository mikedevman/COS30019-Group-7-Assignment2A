# Giải Thích Chi Tiết: Custom Data Preprocessing Pipeline
### File: `data_preproc_custom.py`

---

## So Sánh Nhanh Với `data_preproc_basic.py`

| Khía cạnh | `basic` | `custom` (file này) |
|---|---|---|
| **Mục tiêu model** | LSTM/GRU độc lập từng trạm | **GCN-LSTM / Spatio-Temporal** |
| **Tọa độ địa lý** | Không dùng | ✅ Giữ `lat`, `lon`, lọc Null Island |
| **Đồ thị không gian** | Không có | ✅ KNN adjacency matrix |
| **Tensor đầu ra** | 3D `(samples, seq, features)` | ✅ 4D `(samples, seq, nodes, features)` |
| **Feature thời gian** | Binary (is_weekend, ...) | ✅ **Cyclic encoding** (sin/cos) + Binary + **Lag features** |
| **Sliding window** | Per-SCATS_ID (độc lập) | **Trên toàn bộ đồ thị** cùng lúc |
| **Thư viện thêm** | — | `sklearn.neighbors.NearestNeighbors` |

---

## Tổng Quan Pipeline

```
Raw CSV (Wide Format + Lat/Lon)
        ↓  [Bước 0] load_data()
    DataFrame thô
        ↓  [Bước 1] reshape_data_sum()
Long Format + Tọa độ hợp lệ + Cyclic Features + Lag Features
        ↓  [Bước 2] build_adjacency_matrix()
Ma trận kề A (N×N) + node_to_idx mapping
        ↓  [Bước 3] build_tensor()
Tensor Không-Thời gian: (T, N, F)
        ↓  [Bước 4] create_st_sequences()
4D Arrays X: (samples, seq_len, N, F)   y: (samples, N)
        ↓  [Bước 5] perform_eda()
Biểu đồ EDA
```

---

## Thư Viện Sử Dụng (Dòng 1-5)

```python
import pandas as pd    # Xử lý bảng dữ liệu
import numpy as np     # Tính toán ma trận/tensor
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns  # Vẽ biểu đồ thống kê
from sklearn.neighbors import NearestNeighbors  # Thuật toán KNN cho đồ thị không gian
```

> **Điểm mới so với `basic`:** `NearestNeighbors` từ scikit-learn được dùng để xây dựng đồ thị láng giềng địa lý giữa các trạm SCATS.

---

## Bước 0: Nạp Dữ Liệu — `load_data()` (Dòng 15-17)

```python
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
```

**Logic trong `__main__` (Dòng 227-233):** Khác với `basic` (dùng fallback path list), file `custom` dùng `__file__` để tính **đường dẫn tuyệt đối** từ vị trí của script:

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../data/SCATS_data.csv')
```

- `os.path.abspath(__file__)` → đường dẫn tuyệt đối của script đang chạy.
- `os.path.dirname(...)` → thư mục chứa script.
- `os.path.join(..., '../data/SCATS_data.csv')` → tính ngược lên 1 cấp, vào thư mục `data/`.

> **Lý do:** Cách này **robust hơn** vì không phụ thuộc vào thư mục làm việc hiện tại (CWD) của terminal — script có thể chạy đúng từ bất kỳ đâu.

---

## Bước 1: Reshape & Feature Engineering — `reshape_data_sum()` (Dòng 19-116)

Đây là hàm lớn nhất, bao gồm nhiều sub-bước. Nó **mở rộng đáng kể** so với phiên bản `basic`.

### 1a. Giữ lại tọa độ địa lý (Dòng 28-32)

```python
keep_cols = ['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date'] + v_cols
df = df[keep_cols].copy()
df.rename(columns={
    'SCATS Number': 'SCATS_ID',
    'NB_LATITUDE': 'lat',
    'NB_LONGITUDE': 'lon'
}, inplace=True)
```

**Điểm khác biệt với `basic`:** Thêm 2 cột `NB_LATITUDE`, `NB_LONGITUDE` — đây là tọa độ GPS của mỗi trạm, cần thiết để xây dựng đồ thị không gian ở Bước 2.

### 1b. Loại bỏ "Null Island" (Dòng 34-44)

```python
coord_eps = 1e-6
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')  # Ép kiểu số, NaN nếu lỗi
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

is_null_island = (df['lat'].abs() <= coord_eps) & (df['lon'].abs() <= coord_eps)

# Tính tọa độ trung bình CHỈ từ các hàng có tọa độ hợp lệ (loại bỏ Null Island)
coords_df = df.loc[~is_null_island].groupby('SCATS_ID')[['lat', 'lon']].mean().reset_index()
df = df.drop(columns=['lat', 'lon'])
```

> **"Null Island" là gì?**
> Là tọa độ `(0.0, 0.0)` — điểm giao của đường xích đạo và kinh tuyến gốc, nằm giữa Đại Tây Dương. Đây là **giá trị mặc định lỗi** khi cảm biến không ghi nhận được tọa độ thực.

**Tại sao nguy hiểm nếu không lọc?**
- Nếu giữ tọa độ `(0,0)`, KNN algorithm sẽ tính khoảng cách từ Châu Úc đến điểm giữa đại dương → tất cả trạm sẽ có khoảng cách gần với nhau một cách sai lệch.
- Biểu đồ visualization sẽ bị scale về một điểm.

**Logic xử lý:**
- `pd.to_numeric(..., errors='coerce')` → ép kiểu an toàn, chuỗi không hợp lệ thành `NaN`.
- `~is_null_island` → lấy các hàng **không phải** Null Island.
- `.groupby().mean()` → lấy **tọa độ trung bình** của trạm (vì cùng một trạm có thể xuất hiện nhiều lần với tọa độ hơi khác nhau do độ chính xác GPS).
- `df.drop(columns=['lat', 'lon'])` → tạm xóa tọa độ khỏi df chính (sẽ merge lại sau khi reshape).

### 1c-1e. Reshape Wide → Long Format (Dòng 47-71)

**Giống hệt với `basic`** — xem `explaination_1.md` để biết chi tiết. Bao gồm:
- Chuẩn hóa Date bằng `.dt.normalize()`
- `pd.melt()` → wide → long
- Tính timestamp từ V00-V95 offset 15 phút
- `.groupby(['SCATS_ID', 'Timestamp']).sum()` → gộp nhiều hướng

### 1f. Ghép lại tọa độ & loại trạm không hợp lệ (Dòng 73-77)

```python
df_long = df_long.merge(coords_df, on='SCATS_ID', how='left')
df_long = df_long.dropna(subset=['lat', 'lon']).copy()
```

- `.merge(..., how='left')` → left join: giữ toàn bộ `df_long`, thêm `lat`/`lon` từ `coords_df`.
- Nếu một trạm chỉ tồn tại với tọa độ `(0,0)` (đã bị lọc ở Bước 1b), sau merge lat/lon sẽ là `NaN` → `.dropna()` sẽ loại bỏ toàn bộ trạm đó.

> **Hiệu ứng:** Các trạm có dữ liệu GPS hoàn toàn không hợp lệ **bị loại khỏi pipeline**, tránh làm nhiễu đồ thị không gian.

### 1g. Cyclic Encoding cho thời gian (Dòng 79-95)

Đây là **điểm nâng cấp quan trọng nhất** so với `basic`. Thay vì dùng giá trị số thô (0-23, 0-6), file `custom` áp dụng **sine-cosine encoding** để biểu diễn tính chu kỳ của thời gian.

**Tại sao cần Cyclic Encoding?**

Nếu dùng giá trị số thô:
- `hour=23` và `hour=0` (nửa đêm → 1h sáng) sẽ có khoảng cách `23` đơn vị.
- Nhưng về mặt giao thông, chúng **rất gần nhau** (đều là đêm khuya).

Với sin/cos:
- `f(23)` và `f(0)` sẽ gần nhau trên vòng tròn đơn vị → mô hình học được tính liên tục.

```python
# Chu kỳ theo giờ (0-23)
h = df_long['hour_of_day'].to_numpy(dtype=np.float64)
df_long['hour_sin'] = np.sin(2 * np.pi * h / 24.0)
df_long['hour_cos'] = np.cos(2 * np.pi * h / 24.0)

# Chu kỳ theo slot 15 phút (0-95)
slot_96 = (df_long['Timestamp'].dt.hour * 4 + df_long['Timestamp'].dt.minute // 15).to_numpy(...)
df_long['slot_sin'] = np.sin(2 * np.pi * slot_96 / 96.0)
df_long['slot_cos'] = np.cos(2 * np.pi * slot_96 / 96.0)

# Chu kỳ theo ngày trong tuần (0-6)
dow = df_long['day_of_week'].to_numpy(dtype=np.float64)
df_long['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
df_long['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
```

**Bảng tổng hợp Cyclic Features:**

| Feature pair | Chu kỳ | Công thức tính slot | Mô tả |
|---|---|---|---|
| `hour_sin`, `hour_cos` | 24 giờ | `hour_of_day` | Vị trí trong ngày theo giờ |
| `slot_sin`, `slot_cos` | 96 slots = 24h | `hour*4 + minute//15` | Vị trí trong ngày theo khoảng 15 phút (chi tiết hơn) |
| `dow_sin`, `dow_cos` | 7 ngày | `day_of_week` | Vị trí trong tuần |

> **Tại sao có cả `hour` lẫn `slot`?** `slot` (15 phút) chính xác hơn `hour` vì dữ liệu giao thông thay đổi nhanh trong 15 phút. Tuy nhiên, `hour` vẫn được giữ để mô hình có thêm tín hiệu tổng thể về giờ trong ngày.

### 1h. Binary Features (Dòng 97-101)

```python
df_long['is_weekend']  = (df_long['day_of_week'] >= 5).astype(int)
df_long['is_rush_hour'] = df_long['hour_of_day'].apply(
    lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)
df_long['is_night']    = df_long['hour_of_day'].apply(
    lambda x: 1 if x >= 22 or x <= 5 else 0)
```

Giống với `basic` — **binary flags** (0/1) bổ sung tín hiệu ngữ cảnh rõ ràng.

### 1i. Lag Features (Dòng 103-111)

Đây là **tính năng hoàn toàn mới** so với `basic`.

```python
df_long.sort_values(['SCATS_ID', 'Timestamp'], inplace=True)
g = df_long.groupby('SCATS_ID', sort=False)['Traffic_Volume']

df_long['traffic_lag_1']  = g.shift(1)   # Lưu lượng 15 phút trước
df_long['traffic_lag_4']  = g.shift(4)   # Lưu lượng 1 giờ trước  (4 × 15 phút)
df_long['traffic_lag_96'] = g.shift(96)  # Lưu lượng 1 ngày trước (96 × 15 phút)

df_long[['traffic_lag_1', 'traffic_lag_4', 'traffic_lag_96']].fillna(0.0)
```

**Lag features là gì?**
- `lag_1`: giá trị lưu lượng của **khoảng 15 phút trước** (tại cùng trạm).
- `lag_4`: giá trị lưu lượng của **1 giờ trước**.
- `lag_96`: giá trị lưu lượng của **cùng thời điểm hôm qua**.

**Tại sao hữu ích?**
- Giao thông có **autocorrelation mạnh**: nếu 15 phút trước đông thì bây giờ thường cũng đông.
- `lag_96` nắm bắt **pattern ngày hôm qua**: Thứ Tư hôm nay thường giống Thứ Tư tuần trước.

**Tại sao phải `.groupby('SCATS_ID').shift()`?**
- `.shift()` trên toàn DataFrame sẽ lấy hàng trước **bất kể trạm nào**.
- `.groupby('SCATS_ID').shift()` đảm bảo lag chỉ lấy trong **cùng một trạm**, không bị "bắc cầu" sang trạm khác.
- `fillna(0.0)` → các hàng đầu tiên của mỗi trạm (không có lag) được điền bằng 0.

### 1j. Sắp xếp lại theo thời gian (Dòng 113-114)

```python
df_long.sort_values(by=['Timestamp', 'SCATS_ID'], inplace=True)
df_long.reset_index(drop=True, inplace=True)
```

> **Quan trọng:** Sau khi xử lý lag (phải sort theo `SCATS_ID`), sắp xếp lại theo `Timestamp` để chuẩn bị cho `build_tensor()` — hàm này cần dữ liệu được nhóm theo mốc thời gian, không phải theo trạm.

**Kết quả trả về của `reshape_data_sum()` trong `custom`:**

| SCATS_ID | Timestamp | Traffic_Volume | lat | lon | day_of_week | hour_of_day | hour_sin | hour_cos | slot_sin | slot_cos | dow_sin | dow_cos | is_weekend | is_rush_hour | is_night | lag_1 | lag_4 | lag_96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 970 | 2006-10-01 00:00 | 120 | -37.8 | 144.9 | 6 | 0 | 0.0 | 1.0 | 0.0 | 1.0 | ... | ... | 1 | 0 | 1 | 0 | 0 | 0 |

---

## Bước 2: Xây Dựng Ma Trận Kề — `build_adjacency_matrix()` (Dòng 118-140)

### Mục đích

Xây dựng **đồ thị không gian** giữa các trạm SCATS dựa trên khoảng cách địa lý, dùng cho lớp Graph Convolution (GCN).

### Chi tiết logic (Dòng 123-140)

```python
nodes = sorted(df_long['SCATS_ID'].unique())          # Danh sách trạm đã sắp xếp
node_to_idx = {n: i for i, n in enumerate(nodes)}     # Dict: SCATS_ID → index số

coords = df_long.groupby('SCATS_ID')[['lat', 'lon']].mean().loc[nodes].values
# coords: mảng (N, 2) — tọa độ trung bình của N trạm
```

**Thuật toán KNN (Dòng 128-129):**

```python
nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
distances, indices = nbrs.kneighbors(coords)
```

- `k+1` vì bản thân mỗi điểm luôn là láng giềng gần nhất của chính nó (index 0).
- `nbrs.kneighbors(coords)` → trả về `k+1` láng giềng gần nhất cho **từng chính điểm trong tập huấn luyện**.

**Xây dựng ma trận A (Dòng 131-138):**

```python
N = len(nodes)
A = np.zeros((N, N))   # Ma trận ban đầu toàn 0

for i in range(N):
    for j_idx in indices[i][1:]:   # Bỏ qua j=0 (chính nó)
        dist = np.linalg.norm(coords[i] - coords[j_idx])  # Khoảng cách Euclidean
        A[i][j_idx] = 1 / (dist + 1e-6)  # Trọng số = nghịch đảo khoảng cách
        A[j_idx][i] = A[i][j_idx]         # Ma trận đối xứng
```

**Ý nghĩa trọng số:**
- Trạm **càng gần nhau** → `dist` nhỏ → weight `1/dist` **càng lớn** → ảnh hưởng lẫn nhau nhiều hơn trong GCN.
- `+ 1e-6` trong mẫu số tránh chia cho 0 nếu hai trạm trùng tọa độ.
- Ma trận **đối xứng** (`A[i][j] = A[j][i]`) → đồ thị vô hướng: ảnh hưởng là hai chiều.

**Trực quan hóa:**

```
Trạm A ─── (weight cao) ─── Trạm B  (gần nhau)
Trạm A ──── (weight thấp) ─── Trạm C  (xa nhau)
Trạm A  ×  Trạm D  (không kết nối — ngoài top-k láng giềng)
```

**Đầu ra:**

```python
return A, node_to_idx
# A: np.array shape (N, N) — ma trận kề có trọng số
# node_to_idx: dict {SCATS_ID: index} — mapping cho tensor
```

---

## Bước 3: Tạo Tensor Không-Thời Gian — `build_tensor()` (Dòng 142-168)

### Mục đích

Chuyển đổi dữ liệu Long Format (2D bảng) thành **Tensor 3D `(T, N, F)`** — cấu trúc dữ liệu chuẩn cho mô hình Spatio-Temporal.

```
T = số mốc thời gian
N = số trạm (nodes)
F = số features
```

### Logic (Dòng 146-168)

```python
times = sorted(df_long['Timestamp'].unique())   # Danh sách T mốc thời gian
nodes = list(node_to_idx.keys())                # Danh sách N trạm
T, N, F = len(times), len(nodes), len(feature_cols)

tensor = np.zeros((T, N, F))   # Khởi tạo tensor toàn 0

# Index hóa df_long theo (Timestamp, SCATS_ID) để lookup nhanh
df_indexed = df_long.set_index(['Timestamp', 'SCATS_ID'])[feature_cols]

for t_idx, t in enumerate(times):      # Duyệt từng mốc thời gian
    try:
        df_t = df_indexed.loc[t]        # Lấy slice tại thời điểm t: shape (N_t, F)
        for node in df_t.index:         # Duyệt từng trạm có dữ liệu tại t
            n_idx = node_to_idx[node]   # SCATS_ID → index số
            tensor[t_idx, n_idx, :] = df_t.loc[node].values
    except KeyError:
        continue                        # Bỏ qua nếu không có dữ liệu tại t
```

**Trực quan hóa tensor:**

```
tensor[t=0, :, :]   ← Ma trận (N×F) tại thời điểm đầu tiên
tensor[t=1, :, :]   ← Ma trận (N×F) tại thời điểm thứ 2
...
tensor[t=T-1, :, :] ← Ma trận (N×F) tại thời điểm cuối

tensor[:, n=0, :]   ← Chuỗi thời gian (T×F) của trạm đầu tiên
tensor[:, n=1, :]   ← Chuỗi thời gian (T×F) của trạm thứ 2
```

**Kỹ thuật tối ưu:** `df_long.set_index(['Timestamp', 'SCATS_ID'])` tạo **MultiIndex** → lookup `df_indexed.loc[t]` chạy bằng hash lookup `O(1)` thay vì scan toàn bộ bảng `O(n)`.

**Trạm thiếu dữ liệu tại một mốc t:** Giá trị trong tensor sẽ là `0` (khởi tạo mặc định) — một dạng **zero-imputation** ngầm định.

---

## Bước 4: Sliding Window Spatio-Temporal — `create_st_sequences()` (Dòng 170-186)

### Điểm khác biệt cốt lõi với `basic`

| | `basic` (`create_lstm_sequences`) | `custom` (`create_st_sequences`) |
|---|---|---|
| **Input** | Long Format 2D | **Tensor 3D `(T, N, F)`** |
| **Sliding window** | Per-SCATS_ID | **Toàn bộ N trạm cùng lúc** |
| **X shape** | `(samples, seq, F)` | **`(samples, seq, N, F)`** |
| **y shape** | `(samples,)` — 1 trạm | **`(samples, N)`** — N trạm |
| **Dự đoán** | Lưu lượng 1 trạm tiếp theo | **Lưu lượng của TẤT CẢ trạm tiếp theo** |

### Logic (Dòng 181-186)

```python
X, y = [], []
T = tensor.shape[0]

for i in range(T - seq_len):
    X.append(tensor[i : i + seq_len])       # Slice (seq_len, N, F)
    y.append(tensor[i + seq_len, :, 0])     # Feature 0 (Traffic_Volume) của TẤT CẢ N trạm

return np.array(X), np.array(y)
# X: (T - seq_len, seq_len, N, F)
# y: (T - seq_len, N)
```

**Trực quan hóa:**

```
Tensor:  [t0][t1][t2]...[t95][t96][t97]...[tT]
                                ↑ mỗi là ma trận (N×F)

Window 1: X = tensor[0:96]   (96, N, F)  →  y = tensor[96, :, 0]  (N,)
Window 2: X = tensor[1:97]   (96, N, F)  →  y = tensor[97, :, 0]  (N,)
Window 3: X = tensor[2:98]   (96, N, F)  →  y = tensor[98, :, 0]  (N,)
```

> **Ý nghĩa:** Mỗi sample X là một "bộ phim" 96 frame về trạng thái **toàn bộ mạng lưới giao thông**. Mô hình học cách dự đoán lưu lượng của **tất cả các nút** trong bước tiếp theo — không phải từng trạm riêng lẻ.

---

## Bước 5: EDA — `perform_eda()` (Dòng 189-223)

**Giống hệt với `basic`** — 3 biểu đồ:
1. Chuỗi thời gian 1 tuần tại 1 trạm.
2. Lưu lượng trung bình theo giờ — Ngày thường vs Cuối tuần.
3. Boxplot phân bố Traffic Volume.

---

## Luồng Thực Thi `__main__` (Dòng 227-247)

```
main()
 │
 ├─► load_data(absolute_path)                   → data (Wide Format)
 │
 ├─► reshape_data_sum(data)                      → df_long (Long + Coords + Cyclic + Lag)
 │    print(df_long.head())
 │    print(df_long.shape)
 │
 └─► [Dừng lại] In thông báo hướng dẫn
      "Vui lòng kích hoạt script train_gcn_lstm.py
       để kiểm tra module đồ thị (Graph Convolution)..."
```

> **Lưu ý:** `build_adjacency_matrix()`, `build_tensor()`, `create_st_sequences()` **không được gọi trong `__main__`** của file này — chúng được import và sử dụng bởi script training chính `train_gcn_lstm.py`.

---

## Toàn Bộ Feature Set Đầu Ra

Sau `reshape_data_sum()`, mỗi hàng trong `df_long` có **18 cột**:

| Nhóm | Cột | Loại |
|---|---|---|
| **ID & Time** | `SCATS_ID`, `Timestamp` | Key |
| **Target** | `Traffic_Volume` | Giá trị dự đoán |
| **Tọa độ** | `lat`, `lon` | Dùng cho đồ thị, không phải input model |
| **Time (raw)** | `day_of_week`, `hour_of_day` | Số nguyên |
| **Cyclic (giờ)** | `hour_sin`, `hour_cos` | Float [-1, 1] |
| **Cyclic (slot 15p)** | `slot_sin`, `slot_cos` | Float [-1, 1] |
| **Cyclic (ngày tuần)** | `dow_sin`, `dow_cos` | Float [-1, 1] |
| **Binary** | `is_weekend`, `is_rush_hour`, `is_night` | 0 hoặc 1 |
| **Lag** | `traffic_lag_1`, `traffic_lag_4`, `traffic_lag_96` | Float |

---

## Tóm Tắt Pipeline Theo Hàm

| Bước | Hàm | Input | Output | Shape đầu ra (ví dụ) |
|---|---|---|---|---|
| 0 | `load_data()` | CSV path | DataFrame thô | `(N_rows, N_cols)` |
| 1 | `reshape_data_sum()` | DataFrame Wide | Long Format đầy đủ | `(T×N_stations, 18)` |
| 2 | `build_adjacency_matrix()` | df_long, k | Ma trận kề A + mapping | `A: (N, N)` |
| 3 | `build_tensor()` | df_long, mapping, features | Tensor không-thời gian | `(T, N, F)` |
| 4 | `create_st_sequences()` | Tensor, seq_len | X, y cho ST-model | `X: (S, seq, N, F)`, `y: (S, N)` |
| 5 | `perform_eda()` | df_long | 3 biểu đồ | — |

---

## Lưu Ý Quan Trọng

> **Không có Normalization/Scaling ở đây.**
> Scaling (ví dụ: `MinMaxScaler`) cần được thực hiện sau khi có tensor, thường trong script `train_gcn_lstm.py`, trước khi đưa vào model.

> **`lat`/`lon` KHÔNG nằm trong `feature_cols` khi gọi `build_tensor()`.**
> Chúng chỉ dùng để xây dựng đồ thị adjacency, sau đó không đưa vào tensor features — tránh mô hình học vẹt vị trí địa lý thay vì học pattern giao thông.

> **Tính đồng bộ thời gian (ghost nodes).**
> Nếu một trạm không có dữ liệu tại một mốc t nào đó, `build_tensor()` sẽ để giá trị là `0` (từ `np.zeros`). Đây là một assumption ngầm định — cần kiểm tra tỷ lệ missing data thực tế.

> **Pipeline này tạo ra input cho mô hình GCN-LSTM / ST-GCN.**
> Luồng từ X shape `(S, seq, N, F)` với ma trận kề A sẽ được xử lý: GCN học spatial dependencies → LSTM học temporal dependencies.
