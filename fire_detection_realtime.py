"""
🔥 Real-Time Fire & Smoke Detection
    YOLO (yolo26m best.pt) + Groq Llama-3 AI Alerts
────────────────────────────────────────────────────
Install deps:
    pip install ultralytics groq opencv-python

Run:
    python fire_detection_realtime.py

Controls:
    Q  →  quit
    S  →  save current frame as screenshot
    R  →  force Groq AI analysis right now
    M  →  toggle mute (suppress on-screen AI message)
"""

import cv2
import time
import os
import threading
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from groq import Groq

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────

MODEL_PATH   = "Models/best-2.pt"                           # path to your trained weights
GROQ_API_KEY = "YOUR_API_KEY"                               # get free key at console.groq.com
GROQ_MODEL   = "llama-3.3-70b-versatile"                    # fast and free; options below:
                                                            # "llama3-70b-8192"  (smarter, slower)
                                                            # "llama-3.1-8b-instant" (fastest)

# Detection settings
CONF_THRESHOLD   = 0.40    # minimum confidence to count as detection
ALERT_FRAMES     = 3       # consecutive frames needed before triggering alert
LLM_COOLDOWN_SEC = 8       # minimum seconds between Groq calls
DANGER_WINDOW    = 30      # frames to look back for danger level calculation

# Display
WINDOW_NAME = "🔥 Fire & Smoke Detection — press Q to quit"
SAVE_DIR    = "screenshots"

# Class names from your notebook: names: ['smoke', 'fire']
CLASS_NAMES  = {0: "smoke", 1: "fire"}
CLASS_COLORS = {0: (30, 160, 255), 1: (0, 50, 255)}   # BGR: orange-ish, red


# ─────────────────────────────────────────────────────────────
#  GROQ CLIENT
# ─────────────────────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)

def call_groq(scene_summary: str) -> str:
    """Send a scene description to Groq Llama-3 and return the response."""
    prompt = f"""You are an emergency fire safety AI assistant monitoring a live camera feed.

Current detection summary:
{scene_summary}

Respond in exactly this format — no extra text:
DANGER: [LOW / MEDIUM / HIGH / CRITICAL]
ACTION: [One clear sentence — what should people do right now?]
DETAIL: [One sentence — additional context or warning.]"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"DANGER: UNKNOWN\nACTION: AI unavailable — follow standard fire procedure.\nDETAIL: Error: {e}"


# ─────────────────────────────────────────────────────────────
#  ALERT SYSTEM
# ─────────────────────────────────────────────────────────────

class AlertSystem:
    def __init__(self):
        self.consecutive    = {"fire": 0, "smoke": 0}
        self.last_llm_call  = 0
        self.last_ai_msg    = ""
        self.ai_msg_time    = 0
        self.ai_msg_ttl     = 7       # seconds to show AI message on screen
        self.llm_running    = False   # prevent concurrent calls
        self.history        = deque(maxlen=DANGER_WINDOW)
        self.total_alerts   = 0
        self.muted          = False

    def update(self, detections: list, frame_num: int) -> bool:
        """Update state with current frame detections. Returns True if LLM was triggered."""
        classes_now = {d["class"] for d in detections if d["conf"] >= CONF_THRESHOLD}

        for cls in ["fire", "smoke"]:
            self.consecutive[cls] = (
                self.consecutive[cls] + 1 if cls in classes_now else 0
            )

        has_detection = bool(classes_now)
        self.history.append(has_detection)

        now = time.time()
        triggered = False

        fire_alert  = self.consecutive["fire"]  >= ALERT_FRAMES
        smoke_alert = self.consecutive["smoke"] >= ALERT_FRAMES
        cooldown_ok = (now - self.last_llm_call) > LLM_COOLDOWN_SEC

        if (fire_alert or smoke_alert) and cooldown_ok and not self.llm_running:
            self.last_llm_call = now
            self.total_alerts += 1
            triggered = True

            # Build scene summary for Groq
            fire_conf  = max((d["conf"] for d in detections if d["class"] == "fire"),  default=0)
            smoke_conf = max((d["conf"] for d in detections if d["class"] == "smoke"), default=0)
            n_boxes    = len(detections)
            recent_pct = round(sum(self.history) / max(len(self.history), 1) * 100)

            scene_summary = (
                f"- Fire detected: {'YES' if fire_alert else 'no'}  "
                f"(max confidence: {fire_conf:.0%})\n"
                f"- Smoke detected: {'YES' if smoke_alert else 'no'}  "
                f"(max confidence: {smoke_conf:.0%})\n"
                f"- Total bounding boxes in frame: {n_boxes}\n"
                f"- Detection present in {recent_pct}% of recent frames\n"
                f"- Frame number: {frame_num}"
            )

            # Run Groq in background thread so video stays smooth
            def _run():
                self.llm_running = True
                result = call_groq(scene_summary)
                self.last_ai_msg = result
                self.ai_msg_time = time.time()
                self.llm_running = False
                print(f"\n{'='*55}")
                print(f"[ALERT #{self.total_alerts}] {datetime.now().strftime('%H:%M:%S')}")
                print(scene_summary)
                print("── Groq response ──")
                print(result)
                print('='*55)

            threading.Thread(target=_run, daemon=True).start()

        return triggered

    def danger_level(self) -> str:
        """Quick local danger estimate (before Groq responds)."""
        f = self.consecutive["fire"]
        s = self.consecutive["smoke"]
        if f >= 10 or s >= 15:    return "CRITICAL"
        if f >= 5  or s >= 8:     return "HIGH"
        if f >= ALERT_FRAMES or s >= ALERT_FRAMES: return "MEDIUM"
        if f > 0   or s > 0:      return "LOW"
        return "NONE"

    def active_ai_message(self) -> str | None:
        """Return AI message if still within display TTL, else None."""
        if self.last_ai_msg and (time.time() - self.ai_msg_time) < self.ai_msg_ttl:
            return self.last_ai_msg
        return None


# ─────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────

DANGER_COLORS = {
    "NONE":     (80,  180, 80),    # green
    "LOW":      (50,  200, 200),   # yellow-ish
    "MEDIUM":   (30,  140, 255),   # orange
    "HIGH":     (0,   50,  255),   # red
    "CRITICAL": (0,   0,   255),   # deep red
}

def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls   = d["class"]
        conf  = d["conf"]
        color = CLASS_COLORS.get(cls, (255, 255, 255))

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label   = f"{cls}  {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_hud(frame, alert: AlertSystem, fps: float, frame_num: int):
    H, W = frame.shape[:2]
    danger = alert.danger_level()
    color  = DANGER_COLORS.get(danger, (255, 255, 255))

    # ── Top status bar ──────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Danger badge
    badge_label = f" {danger} "
    (bw, bh), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    cv2.rectangle(frame, (8, 6), (8 + bw + 4, 6 + bh + 6), color, -1)
    cv2.putText(frame, badge_label, (10, 6 + bh + 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    # Fire / smoke consecutive counts
    f_col = CLASS_COLORS[1] if alert.consecutive["fire"]  > 0 else (120, 120, 120)
    s_col = CLASS_COLORS[0] if alert.consecutive["smoke"] > 0 else (120, 120, 120)
    cv2.putText(frame,
                f"fire:{alert.consecutive['fire']}  smoke:{alert.consecutive['smoke']}",
                (bw + 24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # FPS + frame counter (top right)
    info = f"FPS {fps:4.1f}  |  frame {frame_num}"
    (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, info, (W - iw - 8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Groq spinner
    if alert.llm_running:
        dots = "." * (int(time.time() * 3) % 4)
        cv2.putText(frame, f"Groq{dots}", (W - 90, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1)

    # ── AI message box (bottom) ──────────────────────────────
    ai_msg = None if alert.muted else alert.active_ai_message()
    if ai_msg:
        lines      = ai_msg.split("\n")
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        padding    = 10
        line_h     = 22
        box_h      = padding * 2 + line_h * len(lines)

        # Parse DANGER line for color
        box_color = color
        for ln in lines:
            if ln.startswith("DANGER:"):
                level = ln.split(":")[-1].strip()
                box_color = DANGER_COLORS.get(level, color)

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, H - box_h - 4), (W, H), (15, 15, 15), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (0, H - box_h - 4), (4, H), box_color, -1)

        for i, ln in enumerate(lines):
            y = H - box_h - 4 + padding + (i + 1) * line_h
            cv2.putText(frame, ln, (12, y), font, font_scale, (240, 240, 240), thickness)

    # ── Critical flash overlay ───────────────────────────────
    if danger == "CRITICAL":
        if int(time.time() * 2) % 2 == 0:
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (W, H), (0, 0, 180), -1)
            cv2.addWeighted(flash, 0.08, frame, 0.92, 0, frame)
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 220), 4)


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Model loaded ✅")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try changing VideoCapture(0) to VideoCapture(1).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs(SAVE_DIR, exist_ok=True)
    alert     = AlertSystem()
    frame_num = 0
    fps_deque = deque(maxlen=30)
    t_prev    = time.time()

    print(f"\nRunning. Controls: Q=quit  S=screenshot  R=force Groq  M=mute AI messages\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_num += 1

        # ── YOLO inference ─────────────────────────────────
        results    = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id       = int(box.cls[0])
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detections.append({
                "class": CLASS_NAMES.get(cls_id, "unknown"),
                "conf":  conf,
                "bbox":  (x1, y1, x2, y2),
            })

        # ── Alert logic ────────────────────────────────────
        alert.update(detections, frame_num)

        # ── Draw ───────────────────────────────────────────
        draw_detections(frame, detections)

        t_now = time.time()
        fps_deque.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        draw_hud(frame, alert, sum(fps_deque) / len(fps_deque), frame_num)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Key controls ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"fire_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")
        elif key == ord('r'):
            # Force Groq call regardless of cooldown
            alert.last_llm_call = 0
            print("Groq call forced.")
        elif key == ord('m'):
            alert.muted = not alert.muted
            print(f"AI messages {'muted' if alert.muted else 'unmuted'}.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total Groq alerts triggered: {alert.total_alerts}")


if __name__ == "__main__":
    main()
