import numpy as np
import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
import time



df = pd.read_csv('news_dataset_full.csv')


def clean_text(text):
    if isinstance(text, str):  
        text = re.sub(r'<[^>]*>', '', text) 
        text = re.sub(r'\d+', '', text)  
        text = re.sub(r'[^\w\s]', '', text) 
        text = text.lower()  
        return text
    return ''  

# Áp dụng tiền xử lý
df['clean_content'] = df['content'].apply(clean_text)



vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_content'])


encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


configs = [
    {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 5},
    {'learning_rate': 0.0005, 'batch_size': 64, 'epochs': 5},
    {'learning_rate': 0.0001, 'batch_size': 32, 'epochs': 10},
    {'learning_rate': 0.001, 'batch_size': 64, 'epochs': 10}
]


results = []

for config in configs:
    print(f"Training with config: {config}")
    
  
    log_dir = f"logs/fit/{time.strftime('%Y%m%d-%H%M%S')}_lr_{config['learning_rate']}_batch_{config['batch_size']}_epochs_{config['epochs']}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
   
    model = Sequential([
        Dense(512, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(encoder.classes_), activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
   
    history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                        validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
    
   
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    results.append({
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'epochs': config['epochs'],
        'test_loss': test_loss,
        'test_acc': test_acc
    })
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    



results_df = pd.DataFrame(results)
mean_results = results_df.mean()
std_results = results_df.std()

print("\nMean results:")
print(mean_results)

print("\nStandard deviation of results:")
print(std_results)

