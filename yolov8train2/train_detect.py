from ultralytics import YOLO


model = YOLO("yolov8n.pt")

data_yaml = "D:/cashew_yolo/PTUD.v3i.yolov8/data.yaml"

model.train(
    data=data_yaml,
    epochs=50,      
    imgsz=224,        
    batch=8,            
    name="cashew_disease_detect",
    device="cpu",          
    project="runs/detect"  
)

