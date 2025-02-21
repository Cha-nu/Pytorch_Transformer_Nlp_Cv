from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("../models/yolo11x-pose.pt")

# 비디오 추적 실행
results = model.track(
    source="datasets/woman2.mp4",
    show=True,
    save=False
)
