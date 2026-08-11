import cv2
import pickle
import numpy as np
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# CONFIGURATION
# =====================================================

#TARGET_IDENTITY = "Donald_Trump"
#MATCH_THRESHOLD = 0.85

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "known_embeddings.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOG_FILE = os.path.join(LOG_DIR, "detections.txt")

GREEN_THRESHOLD = 0.75
ORANGE_THRESHOLD = 0.85
FRAME_SKIP = 5

# ================= EMAIL CONFIG ======================
EMAIL_ENABLED = True


EMAIL_SENDER = "kommanapavani12@gmail.com"
EMAIL_PASSWORD = "qnhjqdgddzavyucq"   # Gmail App Password
EMAIL_RECEIVER = "rukminivelidi5758@gmail.com"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_COOLDOWN_SECONDS = 60

# ================= INPUT MODE ========================
# Webcam can be triggered via:
# 1) use_webcam=True
# 2) Environment variable USE_WEBCAM=1
USE_WEBCAM = os.getenv("USE_WEBCAM") == "1"

# =====================================================
# INITIAL SETUP
# =====================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,identity,confidence\n")

if not os.path.exists(EMBEDDINGS_FILE):
    print("[ERROR] known_embeddings.pkl not found")
    exit()

# =====================================================
# LOAD MODELS
# =====================================================

print("[INFO] Loading models...")
model = YOLO("detection/weights/best.pt")
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()
print("[INFO] Models loaded successfully.")

# =====================================================
# LOAD EMBEDDINGS
# =====================================================

with open(EMBEDDINGS_FILE, "rb") as f:
    known_embeddings = pickle.load(f)

#if TARGET_IDENTITY not in known_embeddings:
#    print(f"[ERROR] {TARGET_IDENTITY} not in embeddings")
#    exit()

#trump_mean = np.mean(known_embeddings[TARGET_IDENTITY], axis=0)
#print("[INFO] Known embeddings loaded.")

mean_embeddings = {}

for person_name, embeddings in known_embeddings.items():
    mean_embeddings[person_name] = np.mean(embeddings, axis=0)

print("[INFO] Loaded embeddings for:", list(mean_embeddings.keys()))

# =====================================================
# HELPERS
# =====================================================

last_email_time = {}

active_suspects = {}

def log_detection(name, confidence, video_timestamp):
    with open(LOG_FILE, "a") as f:
        f.write(f"{video_timestamp},{name},{confidence:.2f}\n")

"""def match_trump(embedding):
    dist = np.linalg.norm(embedding - trump_mean)
    if dist < MATCH_THRESHOLD:
        return TARGET_IDENTITY, dist
    return "Unknown", dist"""
def match_face(embedding):
    best_name = "Unknown"
    best_distance = float("inf")

    for person_name, mean_embedding in mean_embeddings.items():
        dist = np.linalg.norm(embedding - mean_embedding)

        if dist < best_distance:
            best_distance = dist
            best_name = person_name

    # ---- Convert distance → confidence ----
    confidence = 100 * np.exp(-best_distance * 2)
    confidence = np.clip(confidence, 0, 100)

    return best_name, best_distance, confidence


def send_email_alert(name, image_path, confidence, source, video_timestamp):

    if not EMAIL_ENABLED:
        return

    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        print("[EMAIL WARNING] Email credentials not set")
        return

    now = datetime.now()

    # ---- Cooldown protection ----
    if name in last_email_time:
        if (now - last_email_time[name]).total_seconds() < EMAIL_COOLDOWN_SECONDS:
            return

    last_email_time[name] = now

    try:
        msg = EmailMessage()
        msg["Subject"] = f"🚨 ALERT: {name} Detected"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        # ✅ NEW EMAIL CONTENT
        msg.set_content(
        f"""
        🚨 Suspicious Individual Detected

        Identity   : {name}
        Confidence : {confidence:.2f}%
        Source     : {source}
        Video Time : {video_timestamp}

        Evidence image attached.
        """
                )

        # ---- Attach evidence image ----
        with open(image_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(image_path)
            )

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[EMAIL SENT] Alert sent for {name}")

    except Exception as e:
        print("[EMAIL ERROR]", e)

# =====================================================
# MAIN ANALYSIS (VIDEO + WEBCAM)
# =====================================================

def run_video_analysis(video_path=None, use_webcam=False):

    use_webcam = use_webcam or USE_WEBCAM

    if use_webcam:
        cap = cv2.VideoCapture(0)
        video_source = "Webcam"
        print("[INFO] Using WEBCAM input")
    else:
        video_path = video_path or os.getenv("VIDEO_PATH")
        if not video_path or not os.path.exists(video_path):
            print("[ERROR] Video file not found:", video_path)
            return

        cap = cv2.VideoCapture(video_path)
        video_source = os.path.basename(video_path)
        print("[INFO] Using VIDEO:", video_path)

    frame_count = 0
    last_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            # Draw previous detections so boxes don't blink
            for (x1, y1, x2, y2, name, color, confidence) in last_detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
    frame,
    f"{name} | Confidence: {confidence:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.imshow("Face Recognition (Press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        results = model(frame, verbose=False)
        last_detections = []

        for box in results[0].boxes.xyxy.cpu().numpy():
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = mtcnn(face_rgb)
            if face_tensor is None:
                continue

            if face_tensor.ndim == 4:
                face_tensor = face_tensor[0]

            embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy().flatten()

            name, dist, confidence = match_face(embedding)
            log_detection(name, confidence, video_timestamp)
            # ✅ Get timestamp from VIDEO frame
            video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            seconds = int(video_time_ms / 1000)

            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60

            video_timestamp = f"{hours:02}:{minutes:02}:{secs:02}"

            color = (255, 255, 255)  # default white

# ===============================
# STRONG MATCH (RED ALERT)
# ===============================
            if name != "Unknown" and dist <= GREEN_THRESHOLD:

                color = (0, 0, 255)

    # ✅ Send alert ONLY when suspect newly appears
                if not active_suspects.get(name, False):

                    person_dir = os.path.join(EVIDENCE_DIR, name)
                    os.makedirs(person_dir, exist_ok=True)

                    img_path = os.path.join(
                        person_dir,
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )

                    cv2.imwrite(img_path, face)

                    send_email_alert(
                        name,
                        img_path,
                        confidence,
                        video_source,
                        video_timestamp
                    )

                    active_suspects[name] = True

# ===============================
# BORDERLINE MATCH (ORANGE)
# ===============================
            elif dist <= ORANGE_THRESHOLD:
                color = (0, 165, 255)

                if name in active_suspects:
                    active_suspects[name] = False

                # ===============================
                # UNKNOWN / LEFT FRAME
                # ===============================
            else:
                if name in active_suspects:
                    active_suspects[name] = False


# ---------- DRAW BOX ----------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{name} | Confidence: {confidence:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            last_detections.append(
                (x1, y1, x2, y2, name, color, confidence)
            )
            #-------

        #cv2.imshow("Trump Detection (Press q to quit)", frame)
        cv2.imshow("Face Recognition (Press q to quit)", frame)
        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_video_analysis()
