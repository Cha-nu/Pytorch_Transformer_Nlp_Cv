import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("../models/yolo11x-pose.pt")

def predict(frame, iou=0.7, conf=0.25):
    results = model(
        source = frame,
        device = "0",
        iou=0.7,
        conf=0.25,
        verbose = False
    )
    result = results[0]
    return result

def draw_boxes(result, frame):
    for boxes in result.boxes:
        x1, y1, x2, y2, score, classes = boxes.data.squeeze().cpu().numpy()
        label = f"{model.names[int(score)]} {classes:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 0), 2)
    return frame

def draw_keypoints(result, frame):
    annotator = Annotator(frame, line_width=1)
    for kps in result.keypoints:
        kps = kps.data.squeeze()
        annotator.kpts(kps)
        nkps = kps.cpu().numpy()
        for idx, (x, y, score) in enumerate(nkps):
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)
        return frame


# Load video
capture = cv2.VideoCapture("datasets\woman3.mp4")

if not capture.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Check if the video has reached the last frame
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video

    # Read next frame
    ret, frame = capture.read()
    result = predict(frame)
    frame = draw_boxes(result, frame)
    frame = draw_keypoints(result, frame)
    if not ret:
        print("Error: Couldn't read the frame.")
        break

    # ✅ 프레임 크기 조절 (가로 640, 세로 720으로 변경)
    frame_resized = cv2.resize(frame, (640, 720))

    cv2.imshow("VideoFrame", frame_resized)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
