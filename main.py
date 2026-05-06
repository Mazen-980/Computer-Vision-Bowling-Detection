import os

import cv2
from collections import deque
from ultralytics import YOLO

# =========================
# Input / Output
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(PROJECT_ROOT, "input", "TEST_VIDEO.MOV")
output_path = os.path.join(PROJECT_ROOT, "output", "output_final_car22.avi")
model_path = os.path.join(PROJECT_ROOT, "models", "new_data_best.pt")

# =========================
# Load model
# =========================
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(model_path)

print("Classes:", model.names)

# =========================
# Class IDs (حسب الموديل)
# =========================
BALL = 0
CAR = 1
FALLEN = 2
STANDING = 3

# =========================
# Thresholds
# =========================
CONF_THRESHOLDS = {
    BALL: 0.2,
    CAR: 0.3,
    FALLEN: 0.3,
    STANDING: 0.5
}

IMPACT_MARGIN = 20


def boxes_close(box_a, box_b, margin=0):
    return not (
        box_a[2] < box_b[0] - margin
        or box_a[0] > box_b[2] + margin
        or box_a[3] < box_b[1] - margin
        or box_a[1] > box_b[3] + margin
    )


def get_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def center_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


# =========================
# Colors
# =========================
COLORS = {
    BALL: (0, 140, 255),      # Orange
    CAR: (255, 255, 0),       # Cyan
    FALLEN: (0, 0, 255),      # Red
    STANDING: (0, 200, 0)     # Green
}

input_dir = os.path.dirname(output_path)
if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"Output folder not found: {input_dir}")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found or failed to open: {input_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30

print(f"Video info: {width}x{height} @ {fps:.2f} FPS")

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise SystemExit(
        f"ERROR: Failed to open VideoWriter for {output_path}. "
        "Check codec support and output folder permissions."
    )

# =========================
# Loop
# =========================
frame_count = 0
impact_started = False

# --- Tracking state ---
car_path = deque(maxlen=500)          # car centre positions
# pin_tracker: {id: {center, last_box, state, fall_order, ever_standing, consec_fallen}}
pin_tracker = {}
next_pin_id = 0
fall_order_counter = 0
PIN_MATCH_DIST = 120        # px radius to re-identify pins across frames
FALLEN_CONFIRM = 3          # consecutive FALLEN frames needed to assign an order

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}", end="\r")

    annotated = frame.copy()

    # counters
    standing_count = 0
    fallen_count = 0
    ball_boxes = []
    car_boxes = []
    pin_boxes = []
    pin_detections = []   # (x1,y1,x2,y2, cls_id) for all pins this frame

    # =========================
    # Inference
    # =========================
    results = model(frame, conf=0.2, iou=0.5, verbose=False)[0]

    if results.boxes is not None:
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            cls_id = int(cls_id)
            conf = float(conf)

            # skip low confidence
            if cls_id in CONF_THRESHOLDS:
                if conf < CONF_THRESHOLDS[cls_id]:
                    continue
            else:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())

            # =========================
            # Label + Count
            # =========================
            if cls_id == BALL:
                label = f"Ball {conf:.2f}"
                ball_boxes.append((x1, y1, x2, y2))

            elif cls_id == CAR:
                label = f"Car {conf:.2f}"
                car_boxes.append((x1, y1, x2, y2))
                car_path.append(((x1 + x2) // 2, (y1 + y2) // 2))

            elif cls_id == FALLEN:
                fallen_count += 1
                label = f"Fallen {conf:.2f}"
                pin_boxes.append((x1, y1, x2, y2))
                pin_detections.append((x1, y1, x2, y2, FALLEN))

            elif cls_id == STANDING:
                standing_count += 1
                label = f"Standing {conf:.2f}"
                pin_boxes.append((x1, y1, x2, y2))
                pin_detections.append((x1, y1, x2, y2, STANDING))

            else:
                continue

            color = COLORS[cls_id]

            # draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # draw label
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA
            )

    # =========================
    # Stable score logic
    # =========================
    # Impact = car (or ball) touches any pin
    trigger_boxes = car_boxes + ball_boxes
    if not impact_started and trigger_boxes and pin_boxes:
        for t_box in trigger_boxes:
            if impact_started:
                break
            for pin_box in pin_boxes:
                if boxes_close(t_box, pin_box, IMPACT_MARGIN):
                    impact_started = True
                    break

    # Score = number of confirmed-fallen pins (robust, not noisy per-frame count)
    score = fall_order_counter

    # =========================
    # Car path + Pin fall order
    # =========================
    matched_ids = set()

    for (px1, py1, px2, py2, p_cls) in pin_detections:
        center = get_center((px1, py1, px2, py2))
        best_id, best_dist = None, PIN_MATCH_DIST
        for pid, pdata in pin_tracker.items():
            if pid in matched_ids:
                continue
            d = center_dist(center, pdata["center"])
            if d < best_dist:
                best_dist = d
                best_id = pid

        if best_id is not None:
            matched_ids.add(best_id)
            t = pin_tracker[best_id]
            t["center"] = center
            t["last_box"] = (px1, py1, px2, py2)
            t["state"] = p_cls
            if p_cls == STANDING:
                t["ever_standing"] = True
                t["consec_fallen"] = 0
            else:  # FALLEN
                t["consec_fallen"] = t.get("consec_fallen", 0) + 1
                if (impact_started
                        and t["ever_standing"]
                        and t["fall_order"] is None
                        and t["consec_fallen"] >= FALLEN_CONFIRM):
                    fall_order_counter += 1
                    t["fall_order"] = fall_order_counter
        else:
            # new pin first seen this frame
            pin_tracker[next_pin_id] = {
                "center": center,
                "last_box": (px1, py1, px2, py2),
                "state": p_cls,
                "fall_order": None,
                "ever_standing": p_cls == STANDING,
                "consec_fallen": 1 if p_cls == FALLEN else 0,
            }
            next_pin_id += 1

    # Draw car path (fading trail)
    if len(car_path) >= 2:
        pts = list(car_path)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            cv2.line(annotated, pts[i - 1], pts[i], (255, 255, 0), max(1, int(alpha * 4)))

    # Draw fall-order label on every pin that has been assigned an order
    # (uses last_box so label persists even if pin is not detected this frame)
    for t in pin_tracker.values():
        if t["fall_order"] is not None:
            bx1, by1, _, _ = t["last_box"]
            cv2.putText(
                annotated, f"#{t['fall_order']}",
                (bx1, max(by1 - 28, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
            )

    # =========================
    # HUD
    # =========================
    cv2.putText(annotated, f"Standing: {standing_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.putText(annotated, f"Fallen: {fallen_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(annotated, f"Score: {score}/10", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # write frame
    out.write(annotated)

# =========================
# Cleanup
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nDone! Saved to {output_path}")