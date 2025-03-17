import cv2
import subprocess
from ultralytics import YOLO

# Load YOLOv8 Face Model (Optimized for face detection)
model = YOLO("yolov8n-face.pt")

# Load Gender detection model (OpenCV pre-trained CNN)
gender_proto = "deploy_gender.prototxt"
gender_model = "gender_net.caffemodel"
gender_net = cv2.dnn.readNet(gender_model, gender_proto)
gender_labels = ['Male', 'Female']

# Use Haarcascade for even faster face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get YouTube video URL (FRIENDS sitcom video)
youtube_url = "https://www.youtube.com/watch?v=QdQ58BbGRsA"

# Fetch lower-resolution video stream for faster processing
command = f'yt-dlp -f "best[height<=480]" -g "{youtube_url}"'
stream_url = subprocess.check_output(command, shell=True).decode("utf-8").strip()

if not stream_url:
    print("Unable to fetch stream URL.")
    exit()

cap = cv2.VideoCapture(stream_url)

# Set OpenCV window to fullscreen
cv2.namedWindow("Face Detection & Gender Classification", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Detection & Gender Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_skip = 2  # Skip alternate frames to improve speed
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip processing on alternate frames

    # Resize frame for faster processing
    frame = cv2.resize(frame, (960, 540))

    # Convert frame to grayscale for faster face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    males, females = 0, 0

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        # Prepare image for gender prediction
        blob = cv2.dnn.blobFromImage(cv2.resize(face_crop, (227, 227)), 1.0, (227, 227),
                                    (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_labels[gender_preds[0].argmax()]

        # Count gender
        color = (255, 0, 0) if gender == 'Male' else (255, 20, 147)
        if gender == 'Male':
            males += 1
        else:
            females += 1

        # Draw bounding box on detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display counts on frame
    cv2.putText(frame, f'Males: {males}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Females: {females}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 147), 2)

    # Show the output frame
    cv2.imshow("Face Detection & Gender Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
