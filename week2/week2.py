import tensorflow as tf
import numpy as np
import random
import re
import string
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load dữ liệu IMDb
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(text_ids):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_ids])

random.seed(42)
train_indices = random.sample(range(len(train_data)), 5000)
test_indices = random.sample(range(len(test_data)), 5000)

train_texts = [decode_review(train_data[i]) for i in train_indices]
test_texts = [decode_review(test_data[i]) for i in test_indices]
train_labels = [train_labels[i] for i in train_indices]
test_labels = [test_labels[i] for i in test_indices]

# 2. Tiền xử lý văn bản
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().strip()

train_texts = [preprocess_text(t) for t in train_texts]
test_texts = [preprocess_text(t) for t in test_texts]

# 3. Tokenizer và padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_length = 200
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

configs = [
    # Cấu hình 1: Đơn giản, học nhanh
    {"hidden_layers": 1, "units": 128, "activation": "relu", "optimizer": "adam", "learning_rate": 0.001, "batch_size": 64},

    # Cấu hình 2: Tăng số lớp và units
    {"hidden_layers": 2, "units": 256, "activation": "relu", "optimizer": "adam", "learning_rate": 0.0008, "batch_size": 64},

    # Cấu hình 3: Mạnh hơn, dùng RMSprop cho bài toán NLP
    {"hidden_layers": 2, "units": 512, "activation": "relu", "optimizer": "rmsprop", "learning_rate": 0.0005, "batch_size": 64},

    # Cấu hình 4: Mạng sâu hơn với điều chỉnh learning rate thấp hơn
    {"hidden_layers": 3, "units": 512, "activation": "relu", "optimizer": "adam", "learning_rate": 0.0003, "batch_size": 64},

    # Cấu hình 5: Mạng rất sâu, batch lớn, learning rate thấp
    {"hidden_layers": 4, "units": 512, "activation": "relu", "optimizer": "adam", "learning_rate": 0.0001, "batch_size": 128}
]


# 5. Kết quả
results = []

# 6. Huấn luyện & đánh giá
for idx, cfg in enumerate(configs):
    print(f"\n==============================")
    print(f"Cấu hình {idx+1}: {cfg}")
    
    for run in range(3):
        print(f"Lần chạy {run+1}:", end=' ')
        
        # Tạo model thủ công
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length))
        for _ in range(cfg["hidden_layers"]):
            model.add(tf.keras.layers.Dense(cfg["units"], activation=cfg["activation"]))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Optimizer
        if cfg["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
        elif cfg["optimizer"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=cfg["learning_rate"])
        elif cfg["optimizer"] == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg["learning_rate"])
        else:
            optimizer = cfg["optimizer"]

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Train
        model.fit(np.array(train_padded), np.array(train_labels), epochs=5, batch_size=cfg["batch_size"], verbose=0)

        # Evaluate
        loss, acc = model.evaluate(np.array(test_padded), np.array(test_labels), verbose=0)
        acc_percent = acc * 100
        print(f"Độ chính xác = {acc_percent:.2f}%")

        results.append({
            "Cấu hình": f"Cấu hình {idx+1}",
            "Lần chạy": run + 1,
            "hidden_layers": cfg["hidden_layers"],
            "units": cfg["units"],
            "activation": cfg["activation"],
            "optimizer": cfg["optimizer"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "accuracy_percent": acc_percent
        })

# 7. Lưu kết quả vào CSV
df = pd.DataFrame(results)
df.to_csv("week2/model_results.csv", index=False)

# 8. In tổng kết
print("\n==============================")
print("Tổng hợp độ chính xác trung bình:")
print(df.groupby("Cấu hình")["accuracy_percent"].agg(["mean", "std"]).round(2))
