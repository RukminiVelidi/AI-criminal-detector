import os
import cv2
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# -----------------------------
# INITIALIZE MODELS
# -----------------------------

# YOLOv8 face detection model
model = YOLO("detection/weights/best.pt")
print("YOLOv8 model loaded successfully.")

# MTCNN + FaceNet
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()
print("MTCNN and InceptionResnetV1 models loaded successfully.")

# -----------------------------
# LOAD EXISTING EMBEDDINGS
# -----------------------------

EMBEDDINGS_FILE = "known_embeddings.pkl"

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        known_embeddings = pickle.load(f)
        print("Known embeddings loaded successfully.")
else:
    known_embeddings = {}
    print("No known embeddings found. Starting with an empty dictionary.")

# -----------------------------
# FUNCTION TO GENERATE EMBEDDINGS
# -----------------------------

def save_embeddings_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    for person_name in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"\nProcessing '{person_name}' directory...")
        person_embeddings = []

        for filename in os.listdir(person_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(person_path, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Could not read image: {filename}")
                continue

            print(f"Processing '{filename}'...")

            # YOLO face detection
            results = model(img, verbose=False)

            if results[0].boxes is None:
                print("No face detected by YOLO.")
                continue

            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # Convert to RGB
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # MTCNN face alignment
                try:
                    face_tensor = mtcnn(face_rgb)
                except RuntimeError as e:
                    print("MTCNN failed on this face crop, skipping...")
                    continue

                if face_tensor is None:
                    continue

                # ✅ FIX: add ONLY ONE batch dimension
                face_tensor = face_tensor.unsqueeze(0)  # [1, 3, 160, 160]

                # FaceNet embedding
                embedding = (
                    resnet(face_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

                person_embeddings.append(embedding)

        if person_embeddings:
            known_embeddings[person_name] = person_embeddings
            print(f"Saved {len(person_embeddings)} embeddings for '{person_name}'.")
        else:
            print(f"No valid embeddings found for '{person_name}'.")

    # -----------------------------
    # SAVE EMBEDDINGS
    # -----------------------------
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known_embeddings, f)

    print("\nAll embeddings saved successfully.")

# -----------------------------
# RUN SCRIPT
# -----------------------------

directory_path = "faces_db"
save_embeddings_from_directory(directory_path)
