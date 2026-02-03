import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

VIDEO_PATH = r"C:\Users\ehabillayev.53104\Downloads\Rainy Midtown Manhattan 4K - Driving Downtown - New York City USA - J Utah (1080p, h264).mp4"
CONF_THRES = 0.4
DIST_THRESH = 80
DEVICE = "cpu"

CLASS_COLOR = {
    "car": (148, 0, 211),
    "truck": (148, 0, 211),
    "bus": (148, 0, 211),
    "motorcycle": (148, 0, 211),
    "person": (255, 0, 0),
    "traffic light": (0, 0, 255),
}

TARGET_CLASSES = list(CLASS_COLOR.keys())

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

track_history = defaultdict(list)
vehicle_boxes = {}
selected_id = None
next_id = 1
prev_centers = {}

def mouse_click(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for tid, box in vehicle_boxes.items():
            x1, y1, x2, y2 = map(int, box)
            if x1 < x < x2 and y1 < y < y2:
                selected_id = tid
                if tid not in track_history:
                    track_history[tid] = []
                print(f"[SELECTED] ID {tid}")
                break

cv2.namedWindow("Tracker")
cv2.setMouseCallback("Tracker", mouse_click)

# Hava analizi
def analyze_weather(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness > 160:
        return "Sunny"
    elif 100 < mean_brightness <= 160:
        return "Cloudy"
    else:
        return "Rainy"

# Location analizi (sadə heuristic)
def analyze_location(frame):
    h, w, _ = frame.shape
    upper = frame[0:h//3,:,:]
    edges = cv2.Canny(upper, 50, 150)
    edge_density = np.sum(edges)/edges.size
    if edge_density > 0.07:  # çox kənar → binalar → city
        return "City"
    else:
        return "Highway"

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_THRES, device=DEVICE)

    detections = []
    if results[0].boxes is not None:
        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            cls = results[0].names[int(cls_id)]
            if cls in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                detections.append((cls, (x1,y1,x2,y2), (cx,cy)))

    assigned_ids = set()
    for cls, box, center in detections:
        cx, cy = center
        best_id = None
        min_dist = float('inf')
        for tid, prev in prev_centers.items():
            px, py = prev
            dist = np.hypot(cx - px, cy - py)
            if dist < min_dist and dist < DIST_THRESH:
                min_dist = dist
                best_id = tid
        if best_id is None:
            best_id = next_id
            next_id += 1
        prev_centers[best_id] = center
        assigned_ids.add(best_id)
        vehicle_boxes[best_id] = box

        # draw box
        color = CLASS_COLOR.get(cls,(255,255,255))
        x1,y1,x2,y2 = box
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"ID {best_id} {cls}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # trajectory + arrow only for selected
        if selected_id == best_id:
            track_history[best_id].append(center)
            pts = np.array(track_history[best_id], np.int32)
            if len(pts) > 1:
                cv2.polylines(frame,[pts],False,color,2)
                cv2.arrowedLine(frame, pts[-2], pts[-1], (0,255,0), 3, tipLength=0.4)

    prev_centers = {tid: prev_centers[tid] for tid in assigned_ids}

    # --- WEATHER & LOCATION ---
    weather = analyze_weather(frame)
    location = analyze_location(frame)
    cv2.putText(frame, f"Weather: {weather}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
    cv2.putText(frame, f"Location: {location}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    cv2.imshow("Tracker", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
