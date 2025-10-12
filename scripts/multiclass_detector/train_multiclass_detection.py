from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=r"D:/CliniScan_Project/dataset_final/yolo_dataset_multiclass/data.yaml", # Assumes you run the set.py above
        epochs=20,
        patience=5,
        imgsz=640,
        project=r"D:/CliniScan_Project/yolo_results",
        name="final_model_multiclass",
        device='cpu'
    )
    print("\n✅ Training complete!")

if __name__ == '__main__':
    train_model()
    
    
    
    