# Giải Thích Chi Tiết: Data Preprocessing Pipeline
### File: `data_preproc_basic.py`

---

## Tổng Quan

File này xây dựng một **data preprocessing pipeline hoàn chỉnh** nhằm chuyển đổi dữ liệu giao thông thô từ SCATS (Sydney Coordinated Adaptive Traffic System) thành dạng input chuẩn cho mô hình học sâu LSTM/GRU. Pipeline bao gồm 5 bước chính:

```
Raw CSV (Wide Format)
       ↓  [Bước 0] load_data()
   DataFrame thô
       ↓  [Bước 1] reshape_data_sum()
Long Format + Feature Engineering
       ↓  [Bước 2 & 3 & 4] create_lstm_sequences()
3D Arrays (samples, timesteps, features) → Train / Val / Test
       ↓  [Bước 5] perform_eda()
Biểu đồ phân tích khám phá dữ liệu
```

---

## Thư Viện Sử Dụng (Dòng 1-5)

```python
import pandas as pd    # Xử lý bảng dữ liệu
import numpy as np     # Tính toán ma trận/mảng số
import matplotlib.pyplot as plt  # Vẽ biểu đồ cơ bản
import seaborn as sns  # Vẽ biểu đồ thống kê cao cấp
import os              # Kiểm tra đường dẫn file
```

---

## Bước 0: Nạp Dữ Liệu — `load_data()` (Dòng 15-17)

```python
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
```

**Mục đích:** Đọc file CSV thô vào DataFrame của pandas.

**Dữ liệu thô trông như thế nào (Wide Format):**

| SCATS Number | Date | V00 | V01 | V02 | ... | V95 |
|---|---|---|---|---|---|---|
| 970 | 2006-10-01 | 45 | 38 | 22 | ... | 60 |
| 970 | 2006-10-02 | 51 | 40 | 18 | ... | 75 |

- Mỗi hàng = 1 trạm × 1 ngày.
- Mỗi cột `V00`–`V95` = lưu lượng xe trong 1 khoảng 15 phút tương ứng (**96 khoảng × 15 phút = 24 giờ**).
- Dữ liệu ở dạng **Wide Format** — không phù hợp để mô hình hóa chuỗi thời gian.

**Logic trong `__main__` (Dòng 171-181):** Thử nhiều đường dẫn file khác nhau bằng vòng lặp, cho phép script chạy ở nhiều môi trường làm việc (relative path khác nhau):

```python
data_paths = ['data/data.csv', 'Assignment2B/data/data.csv', '../data/data.csv']
for path in data_paths:
    if os.path.exists(path):
        data = load_data(path)
        break
```

---

## Bước 1: Chuyển Đổi Định Dạng & Feature Engineering — `reshape_data_sum()` (Dòng 19-79)

Đây là bước **trung tâm và phức tạp nhất** của pipeline. Nó thực hiện 3 việc lớn: reshape dữ liệu, xây dựng timestamp chính xác, và tạo các feature mới.

### 1a. Chọn lọc cột (Dòng 24-32)

```python
v_cols = [f'V{i:02d}' for i in range(96)]           # Tạo danh sách ['V00','V01',...,'V95']
v_cols = [col for col in v_cols if col in df.columns] # Lọc những cột thực sự tồn tại trong file

keep_cols = ['SCATS Number', 'Date'] + v_cols
df = df[keep_cols].copy()
df.rename(columns={'SCATS Number': 'SCATS_ID'}, inplace=True)
```

- **List comprehension** `f'V{i:02d}'` tạo tên cột với zero-padding (V00, V01, ..., V09, V10, ...).
- Lọc lại với `if col in df.columns` để tránh `KeyError` nếu dữ liệu thiếu một số cột.
- Đổi tên `SCATS Number` → `SCATS_ID` để tên cột không có khoảng trắng (sạch hơn, lập trình an toàn hơn).

### 1b. Chuẩn hóa ngày (Dòng 34-36)

```python
df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
```

- **Vấn đề:** Một số giá trị trong cột `Date` có thể mang theo giờ dư thừa (ví dụ: `'2006-10-01 00:15:00'`), gây lỗi khi cộng timedelta sau này.
- **Giải pháp:** `.dt.normalize()` cắt phần thời gian, chỉ giữ lại ngày (`2006-10-01 00:00:00`), đảm bảo timestamp được tính toán chính xác tuyệt đối.

### 1c. Reshape từ Wide → Long Format với `.melt()` (Dòng 38-44)

```python
df_long = df.melt(
    id_vars=['SCATS_ID', 'Date'],     # Cột giữ nguyên (làm key)
    value_vars=v_cols,                # Các cột sẽ bị "tan chảy" thành hàng
    var_name='TimeInterval',          # Tên cột định danh khoảng thời gian (V00, V01,...)
    value_name='Traffic_Volume'       # Tên cột giá trị lưu lượng
)
```

**Trực quan hoá `.melt()`:**

```
TRƯỚC (Wide):
SCATS_ID | Date       | V00 | V01 | V95
970      | 2006-10-01 |  45 |  38 |  60

SAU (Long):
SCATS_ID | Date       | TimeInterval | Traffic_Volume
970      | 2006-10-01 | V00          | 45
970      | 2006-10-01 | V01          | 38
970      | 2006-10-01 | V95          | 60
```

Từ **1 hàng** (1 ngày × 1 trạm) → **96 hàng** (96 khoảng 15 phút × 1 trạm).

### 1d. Xây dựng Timestamp chính xác (Dòng 46-52)

```python
minutes_offset = df_long['TimeInterval'].str.replace('V', '').astype(int) * 15
# V00 → 0*15=0 phút, V01 → 1*15=15 phút, V96 → ... (không có)

df_long['TimeDelta'] = pd.to_timedelta(minutes_offset, unit='m')
# Chuyển offset (số phút) thành đối tượng timedelta của pandas

df_long['Timestamp'] = df_long['Date'] + df_long['TimeDelta']
# Cộng ngày + offset → Timestamp đầy đủ
```

**Ví dụ:**
- `V04` → `int("04") * 15 = 60 phút` → `timedelta(minutes=60)` → `2006-10-01 01:00:00`
- `V16` → `16 * 15 = 240 phút` → `timedelta(minutes=240)` → `2006-10-01 04:00:00`

### 1e. Lọc cột & Gộp lưu lượng (Dòng 54-62)

```python
df_long = df_long[['SCATS_ID', 'Timestamp', 'Traffic_Volume']]

# Tổng hợp lưu lượng của TẤT CẢ các hướng (nhánh) tại cùng 1 trạm & 1 thời điểm
df_long = df_long.groupby(['SCATS_ID', 'Timestamp'])['Traffic_Volume'].sum().reset_index()

df_long.sort_values(by=['SCATS_ID', 'Timestamp'], inplace=True)
df_long.reset_index(drop=True, inplace=True)
```

> **Tại sao cần `.groupby().sum()`?**
> Dữ liệu SCATS gốc có thể có nhiều hàng cho **cùng một (SCATS_ID, Timestamp)** vì một giao lộ có nhiều nhánh đường (hướng đi khác nhau). Bước `.groupby().sum()` gộp tất cả các hướng lại thành **tổng lưu lượng thực sự** của trạm đó.

### 1f. Feature Engineering — Tạo thêm đặc trưng thời gian (Dòng 64-78)

| Feature | Code | Logic | Mục đích |
|---|---|---|---|
| `day_of_week` | `.dt.dayofweek` | 0=Thứ 2, 6=Chủ nhật | Phân biệt ngày trong tuần |
| `hour_of_day` | `.dt.hour` | 0–23 | Phân biệt giờ trong ngày |
| `is_weekend` | `lambda x: 1 if x >= 5 else 0` | `day_of_week >= 5` → cuối tuần | Đặc trưng nhị phân |
| `is_rush_hour` | `lambda x: 1 if (6<=x<=9) or (16<=x<=19) else 0` | Sáng 6h-9h và Chiều 16h-19h | Nhận dạng giờ cao điểm |
| `is_night` | `lambda x: 1 if x >= 22 or x <= 5 else 0` | 22h–5h hôm sau | Nhận dạng giờ ban đêm |

Các feature này là **cyclic/categorical time features** — giúp mô hình "được báo trước" về ngữ cảnh thời gian, thay vì phải tự học từ giá trị timestamp thô.

**Kết quả trả về của `reshape_data_sum()`:**

| SCATS_ID | Timestamp | Traffic_Volume | day_of_week | hour_of_day | is_weekend | is_rush_hour | is_night |
|---|---|---|---|---|---|---|---|
| 970 | 2006-10-01 00:00:00 | 120 | 6 | 0 | 1 | 0 | 1 |
| 970 | 2006-10-01 00:15:00 | 95 | 6 | 0 | 1 | 0 | 1 |

---

## Bước 2, 3 & 4: Tạo Sliding Window & Phân Chia Tập Dữ Liệu — `create_lstm_sequences()` (Dòng 81-130)

### Ý tưởng Sliding Window

LSTM cần dữ liệu ở dạng: **"Cho tôi xem X bước thời gian qua → tôi đoán bước tiếp theo"**.

```
Dữ liệu: [t0, t1, t2, ..., t95, t96, t97, ...]

Cửa sổ 1: Input=[t0..t95]   → Label=t96
Cửa sổ 2: Input=[t1..t96]   → Label=t97
Cửa sổ 3: Input=[t2..t97]   → Label=t98
...
```

### Cấu hình tham số (Dòng 81, 194-196)

```python
seq_length = 96    # 96 khoảng × 15 phút = 24 giờ (1 ngày dữ liệu làm input)
train_ratio = 0.7  # 70% dữ liệu đầu → tập train
val_ratio   = 0.15 # 15% tiếp theo     → tập validation
# 15% còn lại                           → tập test
```

### Chi tiết logic vòng lặp (Dòng 91-130)

```python
feature_cols = ['Traffic_Volume', 'day_of_week', 'hour_of_day', 'is_weekend', 'is_rush_hour', 'is_night']

for scats_id, group in df_long.groupby('SCATS_ID'):
    group = group.sort_values('Timestamp')
    data_values = group[feature_cols].values  # Shape: (n_timesteps, 6)
    
    n = len(data_values)
    train_end = int(n * 0.7)           # Điểm kết thúc tập train
    val_end   = int(n * (0.7 + 0.15)) # Điểm kết thúc tập validation
    
    for i in range(n - seq_length):
        x_seq    = data_values[i : i + seq_length]  # Cửa sổ input: shape (96, 6)
        y_target = data_values[i + seq_length, 0]   # Chỉ lấy cột 0 = Traffic_Volume
        target_idx = i + seq_length                  # Vị trí của nhãn y
        
        if target_idx < train_end:  → X_train, y_train
        elif target_idx < val_end:  → X_val, y_val
        else:                       → X_test, y_test
```

> **Tại sao lặp theo từng `SCATS_ID`?**
> Để **không tạo sliding window bắc cầu** — tức là window input không được chứa dữ liệu từ hai trạm khác nhau. Nếu gộp toàn bộ dữ liệu lại rồi tạo window, cửa sổ cuối cùng của trạm A sẽ "tràn sang" đầu dữ liệu của trạm B — gây nhiễu nghiêm trọng.

> **Tại sao phân chia train/val/test theo `target_idx` chứ không phải `i`?**
> Để đảm bảo **phân chia theo thời gian** (không phải phân chia ngẫu nhiên). Nhãn `y` nằm ở vị trí `target_idx` quyết định window đó thuộc tập nào — tránh **data leakage** từ tương lai vào quá khứ.

### Shape dữ liệu đầu ra

```
X_train: (N_train, 96, 6)  ← N mẫu, 96 timestep, 6 features
y_train: (N_train,)         ← N nhãn Traffic_Volume duy nhất

X_val:   (N_val, 96, 6)
y_val:   (N_val,)

X_test:  (N_test, 96, 6)
y_test:  (N_test,)
```

Đây chính xác là input 3D mà **Keras/PyTorch LSTM/GRU** yêu cầu: `(batch_size, timesteps, features)`.

---

## Bước 5: Phân Tích Khám Phá Dữ Liệu (EDA) — `perform_eda()` (Dòng 133-167)

### Biểu đồ 1: Chuỗi thời gian 1 tuần của 1 trạm (Dòng 136-148)

```python
sample_scats = df_long['SCATS_ID'].unique()[0]
sample_data = df_long[
    (df_long['SCATS_ID'] == sample_scats) &
    (df_long['Timestamp'] < df_long['Timestamp'].min() + pd.Timedelta(days=7))
]
plt.plot(sample_data['Timestamp'], sample_data['Traffic_Volume'], marker='.', linestyle='-')
```

**Mục đích:** Kiểm tra xem dữ liệu có chu kỳ ngày/tuần bình thường không (giao thông sáng-chiều cao điểm, đêm thấp).

### Biểu đồ 2: Lưu lượng trung bình theo giờ — Ngày thường vs Cuối tuần (Dòng 150-159)

```python
sns.lineplot(data=df_long, x='hour_of_day', y='Traffic_Volume', hue='is_weekend', ...)
```

**Mục đích:** So sánh hành vi giao thông giữa ngày đi làm và cuối tuần. Thường thấy:
- Ngày thường: 2 đỉnh rõ ràng (sáng ~8h và chiều ~17h).
- Cuối tuần: Đỉnh dẹt hơn, dịch chuyển muộn hơn.

### Biểu đồ 3: Boxplot phân bố Traffic Volume (Dòng 161-167)

```python
sns.boxplot(x=df_long['Traffic_Volume'])
```

**Mục đích:** Phát hiện **outliers** — các giá trị lưu lượng bất thường (ví dụ: tai nạn, sự kiện đặc biệt). Nếu có outlier nhiều, cần xem xét thêm bước **clip/winsorize** trước khi scale dữ liệu.

---

## Luồng Thực Thi `__main__` (Dòng 171-205)

```
main()
 │
 ├─► [Try paths] load_data()              → data (Wide Format DataFrame)
 │
 ├─► reshape_data_sum(data)               → df_long (Long Format + 5 features)
 │    print(df_long.head())
 │    print(df_long.shape)
 │
 ├─► create_lstm_sequences(df_long,
 │       seq_length=96,
 │       train_ratio=0.7,
 │       val_ratio=0.15)                  → X_train, y_train, X_val, y_val, X_test, y_test
 │    print(shapes of all arrays)
 │
 └─► perform_eda(df_long)                 → 3 biểu đồ matplotlib
```

---

## Tóm Tắt Toàn Bộ Pipeline

| Bước | Hàm | Input | Output | Kỹ thuật chính |
|---|---|---|---|---|
| 0 | `load_data()` | CSV path | DataFrame thô | `pd.read_csv()`, path fallback |
| 1a | `reshape_data_sum()` | DataFrame Wide | DataFrame sau lọc cột | List comprehension, `.copy()` |
| 1b | `reshape_data_sum()` | Date có giờ dư | Date chuẩn hóa | `.dt.normalize()` |
| 1c | `reshape_data_sum()` | Wide format | Long format | `pd.melt()` |
| 1d | `reshape_data_sum()` | Chuỗi V00-V95 | Timestamp chính xác | `str.replace()`, `pd.to_timedelta()` |
| 1e | `reshape_data_sum()` | Long format đa hướng | Long format 1 hàng/trạm/thời điểm | `groupby().sum()` |
| 1f | `reshape_data_sum()` | Timestamp | 5 cột feature | `lambda`, `.dt.dayofweek`, `.dt.hour` |
| 2-4 | `create_lstm_sequences()` | Long format | 3D arrays chia 3 tập | Sliding window, `groupby SCATS_ID`, phân chia theo thời gian |
| 5 | `perform_eda()` | Long format | 3 biểu đồ | `matplotlib`, `seaborn` |

---

## Các Điểm Quan Trọng & Lưu Ý

> **Không có Normalization/Scaling trong file này!**
> Bước chuẩn hóa giá trị (MinMaxScaler, StandardScaler) **không được thực hiện** ở đây. Điều đó có nghĩa là pipeline này cần được bổ sung scaling trước khi huấn luyện mô hình, hoặc scaling được thực hiện ở file khác (ví dụ: `data_preproc_custom.py`).

> **`seq_length=96` = 1 ngày**
> Với dữ liệu 15 phút/khoảng, 96 bước = 24 giờ. Mô hình nhìn lại **toàn bộ 1 ngày** để dự đoán khoảng 15 phút tiếp theo.

> **Phân chia train/val/test là time-based, không phải random**
> Đây là thực hành bắt buộc với dữ liệu chuỗi thời gian để tránh data leakage — mô hình không được "biết tương lai" trong quá trình training.

> **Mỗi SCATS_ID xử lý độc lập**
> Sliding window không bao giờ vượt qua ranh giới giữa 2 trạm khác nhau, đảm bảo tính toàn vẹn của dữ liệu không gian-thời gian.
