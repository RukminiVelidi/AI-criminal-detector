import streamlit as st
import os
import subprocess
import sys
import shutil
from datetime import datetime

PROJECT_ROOT = os.getcwd()
FACES_DB = os.path.join(PROJECT_ROOT, "faces_db")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")

PYTHON_EXEC = sys.executable

os.makedirs(FACES_DB, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

st.set_page_config(page_title="Face Recognition Dashboard", layout="wide")
st.title("🎯 Face Recognition Surveillance Dashboard")
st.markdown("YOLOv8 + FaceNet | Multi-Person Dynamic Detection")

st.sidebar.header("Controls")

run_embeddings = st.sidebar.button("🔄 Generate Embeddings")
run_analysis = st.sidebar.button("▶ Run Video Analysis")
reset = st.sidebar.button("🧹 Reset Data")

# -----------------------------
# RESET SYSTEM
# -----------------------------
if reset:
    shutil.rmtree(FACES_DB, ignore_errors=True)
    shutil.rmtree(VIDEOS_DIR, ignore_errors=True)
    os.makedirs(FACES_DB)
    os.makedirs(VIDEOS_DIR)
    st.success("All data reset successfully")

# -----------------------------
# ADD PERSON NAME INPUT
# -----------------------------
st.header("1️⃣ Add Suspect")

person_name = st.text_input("Enter suspect name")

uploads = st.file_uploader(
    "Upload suspect images",
    accept_multiple_files=True,
    type=["jpg", "png", "jpeg"]
)

if person_name:
    person_dir = os.path.join(FACES_DB, person_name.replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)

    if uploads:
        for img in uploads:
            with open(os.path.join(person_dir, img.name), "wb") as f:
                f.write(img.getbuffer())
        st.success(f"{len(uploads)} images uploaded for {person_name}")

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
st.header("2️⃣ Upload Video")

uploaded_video = st.file_uploader("Upload MP4", type=["mp4"])
video_path = None

if uploaded_video:
    video_path = os.path.join(
        VIDEOS_DIR, f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
if run_embeddings:
    st.info("Generating embeddings for all suspects...")
    subprocess.run([PYTHON_EXEC, "generate_embedding.py"])
    st.success("Embeddings generated successfully")

# -----------------------------
# RUN ANALYSIS
# -----------------------------
if run_analysis:
    if not video_path:
        st.error("Upload a video first")
    else:
        env = os.environ.copy()
        env["VIDEO_PATH"] = video_path
        subprocess.Popen([PYTHON_EXEC, "face_recognition.py"], env=env)
        st.success("Analysis started")
        st.warning("Press **q inside the video window** to stop")

st.markdown("---")
st.markdown("**Status:** Multi-person detection enabled 🚀")