import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

# 1. Dữ liệu và tiền xử lý
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# 3. Đánh giá
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
    return correct / len(loader.dataset)

# 4. Huấn luyện
def train(model, optimizer, criterion, train_loader, val_loader, device, config):
    model.to(device)
    for epoch in range(config.epochs):
        model.train()
        total_loss, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()

        train_acc = correct / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader.dataset),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

# 5. Huấn luyện với 40 epochs
def run_with_config(lr, batch_size, opt_name, run_name):
    wandb.init(
        project="cnn-cifar10-basic",
        name=run_name,
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": opt_name,
            "epochs": 40
        },
        reinit=True
    )
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer")

    train(model, optimizer, criterion, train_loader, val_loader, device, config)

    test_acc = evaluate(model, test_loader, device)
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()

# 6. Chạy
run_with_config(lr=0.001, batch_size=64, opt_name="adam", run_name="CNN-Basic-40Epochs")
