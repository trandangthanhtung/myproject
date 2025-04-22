# Cài đặt & Import thư viện
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import os

# 1. Tải & xử lý dữ liệu
data = fetch_california_housing()
X, y = data.data, data.target

# Kiểm tra dữ liệu thiếu
print(f"Dữ liệu thiếu:\n{pd.DataFrame(X).isnull().sum()}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Hàm xây dựng mô hình MLP
def build_model(hidden_layers=[64, 32], learning_rate=0.001, input_shape=(8,)):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))  # đầu ra dự đoán giá (1 số thực)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='mse',
                  metrics=['mae'])  # MAE = Mean Absolute Error
    return model

# 3. Hàm huấn luyện & đánh giá mô hình
def run_experiment(config_id, config, n_runs=5, epochs=50, batch_size=32):
    errors = []
    for run in range(n_runs):
        log_dir = f"logs/config_{config_id}/run_{run}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir)

        model = build_model(hidden_layers=config['hidden_layers'],
                            learning_rate=config['learning_rate'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.1, verbose=0, callbacks=[tensorboard_cb])

        mse, mae = model.evaluate(X_test, y_test, verbose=0)
        errors.append(mae)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"Config {config_id}: MAE trung bình = {mean_error:.4f}, Độ lệch chuẩn = {std_error:.4f}")
    return mean_error, std_error

# 4. Các cấu hình siêu tham số
configs = {
    1: {'hidden_layers': [64, 32], 'learning_rate': 0.001},
    2: {'hidden_layers': [128, 64, 32], 'learning_rate': 0.0005},
    3: {'hidden_layers': [256, 128], 'learning_rate': 0.0001},
    4: {'hidden_layers': [32, 16], 'learning_rate': 0.005},
    5: {'hidden_layers': [64, 64, 64], 'learning_rate': 0.001}
}

# 5. Chạy thử với từng cấu hình
results = []
for config_id, config in configs.items():
    mean_error, std_error = run_experiment(config_id, config)
    results.append((config_id, mean_error, std_error))

# 6. Tổng hợp kết quả
print("\nTổng hợp kết quả:")
for config_id, mean, std in results:
    print(f"Config {config_id}: MAE = {mean:.4f} ± {std:.4f}")
