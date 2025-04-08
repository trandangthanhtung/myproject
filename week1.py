import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Đọc dữ liệu MNIST từ tệp
train_images = idx2numpy.convert_from_file('./data/MNIST/raw/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('./data/MNIST/raw/train-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./data/MNIST/raw/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./data/MNIST/raw/t10k-labels-idx1-ubyte')

# Chuẩn hóa dữ liệu ảnh về khoảng [0, 1]
X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
X_test = test_images.reshape(test_images.shape[0], -1) / 255.0

# Chuyển đổi nhãn sang dạng one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(train_labels.reshape(-1, 1))
y_test = encoder.transform(test_labels.reshape(-1, 1))

# Hiển thị 25 hình ảnh mẫu từ tập huấn luyện
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()

# Vẽ 25 hình ảnh
for i in range(25):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].set_title(f'Label: {train_labels[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Định nghĩa các hàm kích hoạt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Định nghĩa đạo hàm của các hàm kích hoạt
def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Lớp DNN định nghĩa mô hình mạng neuron nhân tạo
class DNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', learning_rate=0.1):
        self.learning_rate = learning_rate
        self.activation = activation
        # Khởi tạo trọng số và bias cho lớp ẩn và lớp đầu ra
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    # Lan truyền tiến (Forward Propagation)
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1) if self.activation == 'relu' else sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    # Lan truyền ngược (Backward Propagation) để cập nhật trọng số
    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (relu_derivative(self.Z1) if self.activation == 'relu' else sigmoid_derivative(self.A1))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Cập nhật trọng số bằng Gradient Descent
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    # Hàm tính mất mát (Cross-Entropy Loss)
    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    
    # Hàm huấn luyện mô hình
    def train(self, X, y, epochs=10, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            if epoch % 2 == 0:
                loss = self.compute_loss(self.forward(X), y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Đánh giá độ chính xác trên tập test
    def evaluate(self, X, y):
        predictions = np.argmax(self.forward(X), axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy

# Thử nghiệm với nhiều bộ siêu tham số
hyperparams = [
    (32, 0.1, 16, 'relu'),
    (16, 0.01, 64, 'sigmoid'),
    (64, 0.001, 32, 'relu'),
    (128, 0.05, 128, 'sigmoid'),
    (32, 0.005, 64, 'relu')
]

results = {}

# Huấn luyện và đánh giá mô hình với từng bộ siêu tham số
for batch_size, lr, hidden_size, activation in hyperparams:
    accuracies = []
    for _ in range(5):  # Chạy thử 5 lần mỗi bộ siêu tham số
        model = DNN(input_size=784, hidden_size=hidden_size, output_size=10, activation=activation, learning_rate=lr)
        model.train(X_train, y_train, epochs=10, batch_size=batch_size)
        acc = model.evaluate(X_test, y_test)
        accuracies.append(acc)
    
    # Tính trung bình và độ lệch chuẩn
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    results[(batch_size, lr, hidden_size, activation)] = (mean_acc, std_acc)
    print(f"Siêu tham số {batch_size, lr, hidden_size, activation} -> Độ chính xác: {mean_acc:.4f} ± {std_acc:.4f}")

# Vẽ biểu đồ so sánh kết quả
fig, ax = plt.subplots()
labels = [str(p) for p in results.keys()]
means = [v[0] for v in results.values()]
stdevs = [v[1] for v in results.values()]
ax.barh(labels, means, xerr=stdevs, capsize=5)
ax.set_xlabel("Độ chính xác")
ax.set_title("So sánh các bộ siêu tham số")
plt.show()
