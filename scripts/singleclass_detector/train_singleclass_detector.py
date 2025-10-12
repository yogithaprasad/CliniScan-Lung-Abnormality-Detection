from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8s.pt") # Use the powerful 'small' model
    results = model.train(
        data=r"D:/CliniScan_Project/dataset_final/yolo_dataset_single_class/data.yaml",
        epochs=20,
        patience=5,
        imgsz=640,
        project=r"D:/CliniScan_Project/yolo_results",
        name="final_model_singleclass",
        device='cpu'
    )
    print("\n✅ Training complete!")

if __name__ == '__main__':
    train_model()