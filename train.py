from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

train_results = model.train(
    data = 'D:\yolov11\datasets\data.yaml',
    epochs = 100,  
    imgsz = 640,
    device = '0',  # Use GPU 0, change as needed
    workers = 0
)

metrics = model.val() 
