import cv2
from ultralytics import YOLO

# COCO dataset class names (80 classes)
class_names = [
    "person","bicycle","car","motorbike","aeroplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
    "bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
    "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

video_path = "C:/Users/agane/Desktop/video/video.mp4"  # Change if needed
output_path = "output.avi"

# Load medium YOLOv8 model
model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

# Get video properties for saving output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object to save output
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # or 'MJPG' or 'DIVX'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        # Draw bounding box and label on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"Detected {label} with confidence {conf:.2f}")

    # Write the frame with detections
    out.write(frame)

    # Optional: display live detection
    cv2.imshow("YOLOv8m Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
