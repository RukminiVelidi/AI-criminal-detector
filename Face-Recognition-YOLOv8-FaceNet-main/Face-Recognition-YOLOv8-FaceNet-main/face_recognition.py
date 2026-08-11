from turtle import distance

import cv2
import pickle
import numpy as np
import os
import smtplib
from datetime import datetime
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


MATCH_THRESHOLD = 0.85



EMBEDDINGS_FILE = os.path.join(BASE_DIR, "known_embeddings.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOG_FILE = os.path.join(LOG_DIR, "detections.txt")

FRAME_SKIP = 15

# ================= EMAIL CONFIG ======================
EMAIL_ENABLED = True

EMAIL_SENDER = "kommanapavani12@gmail.com"
EMAIL_PASSWORD = "qnhjqdgddzavyucq"   # Gmail App Password
EMAIL_RECEIVER = "rukminivelidi5758@gmail.com"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_COOLDOWN_SECONDS = 20

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
        f.write("video_time,identity,confidence\n")

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


print("[INFO] Known embeddings loaded.")
# print("Available identities:", known_embeddings.keys())

# =====================================================
# HELPERS
# =====================================================

last_email_time = {}
active_suspects = {}

def log_detection(name, confidence, video_timestamp):
    with open(LOG_FILE, "a") as f:
        f.write(f"{video_timestamp},{name},{confidence:.2f}\n")
def match_identity(embedding):

    min_dist = float("inf")
    best_match = "Unknown"

    for name, embeddings in known_embeddings.items():
        mean_embedding = np.mean(embeddings, axis=0)

        dist = np.linalg.norm(embedding - mean_embedding)

        if dist < min_dist:
            min_dist = dist
            best_match = name

    if min_dist < MATCH_THRESHOLD:
        return best_match, min_dist

    return "Unknown", min_dist
def distance_to_confidence(distance):
    confidence = np.exp(-distance * 2.5)
    return round(confidence * 100, 2)   

def send_email_alert(name, image_path, confidence, source, video_timestamp):

    if not EMAIL_ENABLED:
        return

    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        print("[EMAIL WARNING] Email credentials not set")
        return


    # cooldown protection
    now = datetime.now()

    if name in last_email_time:
        if (now - last_email_time[name]).total_seconds() < EMAIL_COOLDOWN_SECONDS:
            return

    last_email_time[name] = now

    try:
        msg = EmailMessage()
        msg["Subject"] = f"🚨 ALERT: {name} Detected"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        # ✅ UPDATED EMAIL CONTENT
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

        # attach evidence image
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

def run_video_analysis(video_path=None, use_webcam= True):

    use_webcam = use_webcam or USE_WEBCAM
    video_path = "videos/1mininme.mp4"


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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Live Suspect Detection (Press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        results = model(frame, verbose=False)

        for box in results[0].boxes.xyxy.cpu().numpy():
            frame_h, frame_w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

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
            name, dist = match_identity(embedding)
            video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Webcam fallback
            if video_time_ms <= 0:
                video_timestamp = datetime.now().strftime("%H:%M:%S")
            else:
                seconds = int(video_time_ms / 1000)
                h = seconds // 3600
                m = (seconds % 3600) // 60
                s = seconds % 60

                video_timestamp = f"{h:02}:{m:02}:{s:02}"

            confidence = distance_to_confidence(dist)
          
            label = f"{name} ({confidence}%)"

            # Decide box color
            # FINAL COLOR SYSTEM
            if name != "Unknown" and confidence >= 80:
                box_color = (0, 0, 255)      # RED → suspect
                text_color = (255, 255, 255)
            else:
                box_color = (255, 255, 255)  # WHITE → normal
                text_color = (0, 0, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                box_color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )

            # Only save evidence & send email for strong matches
            # STRONG MATCH (GREEN)
            if name != "Unknown" and confidence >= 80:
                log_detection(name, confidence, video_timestamp)
                # If suspect just appeared (not already active)
                if not active_suspects.get(name, False):

                    person_dir = os.path.join(EVIDENCE_DIR, name)
                    os.makedirs(person_dir, exist_ok=True)

                    img_path = os.path.join(
                        person_dir,
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )

                    cv2.imwrite(img_path, face)

                    send_email_alert(name, img_path, confidence, video_source, video_timestamp)

                    active_suspects[name] = True

            else:
                # Reset when suspect disappears
                if name in active_suspects:
                    active_suspects[name] = False
                

            #-------------------------------------------------------------

        cv2.imshow("Live Suspect Detection (Press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_video_analysis()
