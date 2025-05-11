from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load mô hình SOTA nhỏ
model = YOLO('yolov8s.pt')

# Cấu hình 1 - SGD
model.train(
    data='D:/Deep_learning/week5/datasets/cifar10_yolo/data.yaml',
    epochs=30,
    imgsz=64,
    batch=64,
    name='yolo_cifar10_cfg1',
    project='runs/train',
    device='cpu',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    optimizer='SGD'
)

# Cấu hình 2 - Adam
model.train(
    data='D:/Deep_learning/week5/datasets/cifar10_yolo/data.yaml',
    epochs=30,
    imgsz=64,
    batch=128,
    name='yolo_cifar10_cfg2',
    project='runs/train',
    device='cpu',
    lr0=0.003,
    momentum=0.9,
    weight_decay=0.0001,
    optimizer='Adam'
)

# Cấu hình 3 - AdamW
model.train(
    data='D:/Deep_learning/week5/datasets/cifar10_yolo/data.yaml',
    epochs=30,
    imgsz=64,
    batch=256,
    name='yolo_cifar10_cfg3',
    project='runs/train',
    device='cpu',
    lr0=0.001,
    momentum=0.95,
    weight_decay=0.0001,
    optimizer='AdamW'
)

