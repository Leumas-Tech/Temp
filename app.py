import math
import time
from collections import deque
from typing import Tuple
import platform
import os

import cv2
from flask import Flask, Response, render_template
import mediapipe as mp

app = Flask(__name__)

# --- MediaPipe Hands (limit to ONE hand) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,            # ONLY ONE HAND
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Neon-ish colors (BGR)
C_CYAN = (255, 255, 0)
C_LIME = (0, 255, 170)
C_ORANGE = (0, 165, 255)
C_WHITE = (245, 245, 245)
C_BG = (0, 0, 0)

fps_hist = deque(maxlen=30)

def angle_deg(a, b, c) -> float:
    ax, ay, az = a; bx, by, bz = b; cx, cy, cz = c
    v1 = (ax - bx, ay - by, az - bz)
    v2 = (cx - bx, cy - by, cz - bz)
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    m1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2) or 1e-6
    m2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2) or 1e-6
    cosang = max(-1.0, min(1.0, dot/(m1*m2)))
    return math.degrees(math.acos(cosang))

def bbox_from_landmarks(landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    xs = [max(0, min(w-1, x)) for x in xs]
    ys = [max(0, min(h-1, y)) for y in ys]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return x1, y1, x2, y2

def draw_neon_box(img, x1, y1, x2, y2, label="Hand"):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), C_BG, thickness=-1)
    img[:] = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
    L = 30; T = 6
    for (xa, ya) in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        x_end = xa + (L if xa == x1 else -L)
        cv2.line(img, (xa, ya), (x_end, ya), C_CYAN, T, cv2.LINE_AA)
        y_end = ya + (L if ya == y1 else -L)
        cv2.line(img, (xa, ya), (xa, y_end), C_CYAN, T, cv2.LINE_AA)
    pad_x, pad_y = 10, 8
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pill_w, pill_h = tw + pad_x*2, th + pad_y*2
    px, py = x1, max(0, y1 - pill_h - 8)
    cv2.rectangle(img, (px, py), (px+pill_w, py+pill_h), C_CYAN, 2, cv2.LINE_AA)
    cv2.rectangle(img, (px, py), (px+pill_w, py+pill_h), (18,18,18), -1)
    cv2.putText(img, label, (px+pad_x, py+pill_h - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

def draw_thumb_index_hud(img, tpt, ipt):
    tx, ty = tpt; ix, iy = ipt
    mid = ((tx+ix)//2, (ty+iy)//2)
    dpx = int(math.hypot(tx-ix, ty-iy))
    r = max(18, min(120, int(dpx * 0.6)))
    cv2.circle(img, mid, r, C_CYAN, 2, cv2.LINE_AA)
    cv2.circle(img, mid, max(5, int(r*0.6)), C_LIME, 2, cv2.LINE_AA)
    ctrl = (mid[0], mid[1] - int(r*0.9))
    steps = 16; prev = (tx, ty)
    for s in range(1, steps+1):
        t = s/steps
        x = int((1-t)**2*tx + 2*(1-t)*t*ctrl[0] + t**2*ix)
        y = int((1-t)**2*ty + 2*(1-t)*t*ctrl[1] + t**2*iy)
        if s % 2 == 0:
            cv2.line(img, prev, (x,y), C_ORANGE, 2, cv2.LINE_AA)
        prev = (x,y)
    cv2.circle(img, (tx,ty), 7, C_ORANGE, -1, cv2.LINE_AA)
    cv2.circle(img, (ix,iy), 7, C_LIME, -1, cv2.LINE_AA)

# ------- Robust camera opener (Windows prefers DirectShow) -------
def open_camera():
    """
    Try several backends and indices, set friendly properties.
    Returns an opened cv2.VideoCapture or raises RuntimeError.
    """
    # Optional: tell OpenCV to de-prioritize MSMF globally
    if platform.system() == "Windows":
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    candidates = []
    if platform.system() == "Windows":
        candidates = [
            (cv2.CAP_DSHOW, [0,1,2,3]),     # DirectShow first
            (cv2.CAP_MSMF,  [0,1,2,3]),     # then MediaFoundation
            (cv2.CAP_ANY,   [0,1,2,3])      # then "any"
        ]
    else:
        candidates = [(cv2.CAP_ANY, [0,1,2,3])]

    for api, indices in candidates:
        for idx in indices:
            cap = cv2.VideoCapture(idx, api)
            if not cap.isOpened():
                cap.release(); continue
            # Try to set sane properties (some backends ignore these, that’s ok)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # test grab
            ok, _ = cap.read()
            if ok:
                print(f"[Camera] Opened index {idx} via API {api}")
                return cap
            cap.release()
    raise RuntimeError("Could not open any camera (tried DirectShow/MSMF/Any on indices 0..3).")

def gen_frames():
    cap = open_camera()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Attempt to recover: reopen the camera once
                try:
                    cap.release()
                except:
                    pass
                time.sleep(0.25)
                try:
                    cap = open_camera()
                    continue
                except Exception as re:
                    print("[Camera] Reopen failed:", re)
                    time.sleep(0.5)
                    continue

            tic = time.time()
            # Selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            toc = time.time()
            inst_fps = 1.0 / max(1e-6, toc - tic)
            fps_hist.append(inst_fps)
            fps = sum(fps_hist) / len(fps_hist)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                lm = hand_landmarks.landmark

                x1, y1, x2, y2 = bbox_from_landmarks(lm, w, h)
                handed = "Hand"
                if result.multi_handedness:
                    handed = result.multi_handedness[0].classification[0].label
                draw_neon_box(frame, x1, y1, x2, y2, f"{handed}")

                # Skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # Fingertip highlights + HUD for thumb–index
                tip_idx = [4, 8, 12, 16, 20]
                for idx in tip_idx:
                    cx, cy = int(lm[idx].x*w), int(lm[idx].y*h)
                    cv2.circle(frame, (cx, cy), 6, C_WHITE, -1, cv2.LINE_AA)

                t4 = (lm[4].x, lm[4].y, lm[4].z)
                i8 = (lm[8].x, lm[8].y, lm[8].z)
                wrist = (lm[0].x, lm[0].y, lm[0].z)
                t4p = (int(lm[4].x*w), int(lm[4].y*h))
                i8p = (int(lm[8].x*w), int(lm[8].y*h))
                draw_thumb_index_hud(frame, t4p, i8p)

                dist = math.sqrt((t4[0]-i8[0])**2 + (t4[1]-i8[1])**2 + (t4[2]-i8[2])**2)
                ang = angle_deg(t4, wrist, i8)
                panel = f"FPS: {fps:.1f} | {handed} | Thumb-Index: {dist:.4f} | Angle: {ang:.1f}°"
            else:
                panel = f"FPS: {fps:.1f} • No hands"

            # Overlay telemetry bar
            bar_w = min(10 + len(panel)*9, w-20)
            cv2.rectangle(frame, (10, h-40), (10+bar_w, h-10), (20,20,20), -1, cv2.LINE_AA)
            cv2.putText(frame, panel, (16, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue
            jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    finally:
        try:
            cap.release()
        except:
            pass
        hands.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
