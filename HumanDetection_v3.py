import streamlit as st
import cv2
import os
import subprocess
import tempfile
import time
from ultralytics import YOLO
import numpy as np

# Set up YOLOv8 Face Model
yolo_model = YOLO("yolov8n-face.pt")

# Streamlit UI Header
st.title("AI-Powered Face Detection from YouTube Videos")

# User Inputs YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:")


def download_youtube_video(url):
    """Downloads YouTube video and returns the local file path."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "video.mp4")

    command = f'yt-dlp -f "best[height<=720]" -o "{video_path}" "{url}"'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        st.error("Failed to download video. Please check the URL.")
        return None

    return video_path


if st.button("Process Video"):
    if youtube_url:
        st.write("Downloading video...")
        video_path = download_youtube_video(youtube_url)

        if video_path:
            st.success("Video downloaded successfully!")
            cap = cv2.VideoCapture(video_path)
            frame_skip = 2  # Skip alternate frames for speed
            frame_count = 0
            temp_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS) / frame_skip)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = yolo_model(frame_rgb)

                for result in results:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)

                        # Draw bounding box around detected face
                        color = (0, 255, 0)  # Green color for face bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                out.write(frame)

            cap.release()
            out.release()

            st.success("Processing complete! Playing processed video...")
            st.video(temp_video_path)
    else:
        st.warning("Please enter a valid YouTube URL.")
