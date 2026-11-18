
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    # 사전학습된 YOLOv8s-Seg 모델 로드
    model = YOLO("yolov8s-seg.pt")

    # GPU 학습
    model.train(
        data="data.yaml",
        epochs=3,
        imgsz=960,
        batch=4,
        optimizer="AdamW",
        lr0=0.00015,
        mosaic=0.0,
        mixup=0.0,
        patience=20,
        seed=42,
        device=0,
        workers=0,
        project="runs",
        name="step3"
    )


if __name__ == "__main__":
    freeze_support()
    main()
