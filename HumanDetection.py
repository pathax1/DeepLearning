import cv2
import subprocess
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # YOLOv8 nano version (faster, lightweight)

# YouTube live stream URL
youtube_url = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"

# Get the direct stream URL using yt-dlp
command = f'yt-dlp -g "{youtube_url}"'
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stream_url, error = process.communicate()
stream_url = stream_url.decode("utf-8").strip()

if not stream_url:
    print("Failed to fetch live stream URL")
    exit()

# Open video stream using OpenCV
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO detection
    results = model(frame)

    human_count = 0  # Counter for detected humans

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:  # Class ID 0 corresponds to 'person'
                human_count += 1  # Increment human count

                # Draw bounding box around detected person
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display human count on the video frame
    cv2.putText(frame, f"Humans detected: {human_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("Live YouTube Stream - Human Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
