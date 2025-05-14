import gradio as gr
from PIL import Image
from ultralytics import YOLO

# Load model đã huấn luyện
model = YOLO("runs/train/yolo_cifar10_cfg2/weights/best.pt")

def detect_objects(image, conf=0.25):
    # Chạy suy luận
    results = model(image, conf=conf)
    # Lấy ảnh kết quả từ YOLO (vẽ bounding boxes)
    result_img = results[0].plot()  # numpy array
    return Image.fromarray(result_img)

# Giao diện
demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(0, 1, value=0.25, label="Confidence Threshold")
    ],
    outputs=gr.Image(type="pil"),
    title="YOLOv8 CIFAR-10 Object Detection",
    description="Upload an image to detect CIFAR-10 objects using a trained YOLOv8 model."
)

if __name__ == "__main__":
    demo.launch()
