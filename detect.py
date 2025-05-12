from ultralytics import YOLO

model = YOLO('best.pt') 

results = model('test/t5.jpg')
results[0].show()