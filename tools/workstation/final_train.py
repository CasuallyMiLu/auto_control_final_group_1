from ultralytics import YOLO
m = YOLO("work/runs/y26s_det_960/weights/best.pt")
m.train(data="work_final/ds_final/data.yaml", imgsz=960, epochs=60, patience=20, batch=16, amp=True, workers=8, cache="ram",
        cos_lr=True, close_mosaic=10, degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.5, mosaic=1.0, seed=0,
        project="work_final/runs", name="final_y26s_det_960", exist_ok=True, plots=True)
