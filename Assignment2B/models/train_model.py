import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Hàm từ file đã viết trước đó
from data_preproc_basic import load_data, reshape_data_sum, create_lstm_sequences

def build_lstm_model(input_shape):

    """
    Xây dựng một mạng LSTM cơ bản
    """

    model = Sequential()
    
    # Layer LSTM 1
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Layer LSTM 2
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer Mật độ (Dense) cho output
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def main():
    print("1. Đang tải và tiền xử lý dữ liệu...")
    try:
        data = load_data('data.csv')
    except:
        data = load_data('../data.csv')
        
    df_long = reshape_data_sum(data)
    
    # Dữ liệu LSTM phải được đem đi scale từ 0 đến 1 thì Neural Network mới hội tụ tốt.
    print("2. Đang chuẩn hóa (Scale) đa biến về 0-1...")
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    feature_cols = ['Traffic_Volume', 'day_of_week', 'hour_of_day', 'is_weekend', 'is_rush_hour', 'is_night']
    
    # Ta copy lại giá trị thật để so sánh sau này nếu cần
    df_long['Traffic_Volume_Original'] = df_long['Traffic_Volume']
    
    # Fit scaler riêng cho y (cột số 0) để inverse_transform sau này dễ dàng
    scaler_y.fit(df_long[['Traffic_Volume']])
    
    # Scale tất cả các features đang có
    df_long[feature_cols] = scaler_x.fit_transform(df_long[feature_cols])
    
    # Cửa sổ trượt (Time Steps = 96 tương đương dữ liệu ngày trước đoán ngày sau)
    sequence_length = 96
    print(f"3 & 4. Đang tạo Sequences và chẻ Dataset (Time steps = {sequence_length})...")
    X_train, y_train, X_val, y_val, X_test, y_test = create_lstm_sequences(df_long, seq_length=sequence_length, train_ratio=0.7, val_ratio=0.15)
    
    print(f"X_train size: {X_train.shape}")
    print(f"X_val size: {X_val.shape}")
    print(f"X_test size: {X_test.shape}")
    
    print("\n5. Compile và Train LSTM...")
    # shape cho input là (time_steps, features) cụ thể là (96, 1)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Early stop để chống overfit
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Huấn luyện (giới hạn ở 20 epochs để debug nhanh, có thể tăng lên sau)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n6. Đang vẽ biẻu đồ lịch sử Huấn Luyện (Loss)....")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Traning Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Độ mất mát (Loss) của Model qua các Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print("\n7. Đánh giá Model thực trên tập Test Set...")
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE - scaled format): {test_loss}")
    
    # Dự đoán thử một đoạn trên Test
    predictions = model.predict(X_test)
    
    # Bấm ngược lại scale ra số thật
    predictions_denorm = scaler_y.inverse_transform(predictions)
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Vẽ vài khung so sánh kết quả prediction vs thực tế
    plt.figure(figsize=(15, 6))
    # Hiện 500 điểm dữ liệu đầu trong Test để dễ nhìn
    plt.plot(y_test_denorm[:500], label='Thực tế (Ground Truth)', color='blue')
    plt.plot(predictions_denorm[:500], label='Dự đoán (Predictions)', color='red', alpha=0.7)
    plt.title('Dự đoán lưu lượng giao thông LSTM vs Thực tế')
    plt.xlabel('Thời gian (15-min Intervals) trong test set')
    plt.ylabel('Lưu lượng xe')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
