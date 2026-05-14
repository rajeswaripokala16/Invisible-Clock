"""
Gesture-controlled Invisibility
- Open palm  -> normal (visible)
- Fist       -> invisible (face blurred)
"""

import cv2
import mediapipe as mp

# ----- setup -----
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)  # real-time hand tracking [web:62][web:61]

face_det = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)  # real-time face detection [web:66][web:75]

FIST_THRESHOLD = 0.05  # gesture tuning


def _dist(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def is_fist(landmarks, w, h):
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    norm = _dist(wrist, middle_tip, w, h) / h
    return norm < FIST_THRESHOLD     # small distance → closed hand


def is_open_palm(landmarks, w, h):
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    norm = _dist(wrist, middle_tip, w, h) / h
    return norm > FIST_THRESHOLD * 2  # large distance → open hand


cap = cv2.VideoCapture(0)
invisible = False  # current state

print("Show FIST = invisible, OPEN PALM = visible, press q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- detect face ---
    face_result = face_det.process(rgb)
    face_bbox = None
    if face_result.detections:
        for det in face_result.detections:
            box = det.location_data.relative_bounding_box
            x, y = int(box.xmin * w), int(box.ymin * h)
            bw, bh = int(box.width * w), int(box.height * h)
            face_bbox = (x, y, bw, bh)
            # optional debug box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

    # --- detect hand + gesture ---
    hand_result = hands.process(rgb)
    gesture = None

    if hand_result.multi_hand_landmarks:
        for handLms in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            if is_fist(handLms.landmark, w, h):
                gesture = "FIST"
            elif is_open_palm(handLms.landmark, w, h):
                gesture = "OPEN_PALM"

    if gesture == "FIST":
        invisible = True
    elif gesture == "OPEN_PALM":
        invisible = False

    output = frame.copy()

    # --- apply invisibility effect (blur face only) ---
    if invisible and face_bbox:
        x, y, bw, bh = face_bbox
        x, y = max(0, x), max(0, y)
        bw, bh = max(1, bw), max(1, bh)

        roi = output[y:y + bh, x:x + bw]
        if roi.size != 0:
            # strong blur just on face region [web:88][web:90]
            blurred = cv2.GaussianBlur(roi, (51, 51), 30)
            output[y:y + bh, x:x + bw] = blurred

    # HUD
    state = "INVISIBLE" if invisible else "VISIBLE"
    cv2.putText(output, f"State: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if gesture:
        cv2.putText(output, f"Gesture: {gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Gesture Invisibility", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
