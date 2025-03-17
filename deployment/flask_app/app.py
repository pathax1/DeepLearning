# app.py (Flask App for Deepfake Voice & Face Generation)

from flask import Flask, request, render_template, send_file
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Deepfake Voice & Face Generation Service"

@app.route('/generate', methods=['POST'])
def generate_deepfake():
    if 'image' not in request.files or 'audio' not in request.files:
        return "Missing image or audio file.", 400

    image_file = request.files['image']
    audio_file = request.files['audio']

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)

    image.save(image_path)
    audio.save(audio_path)

    # Call synchronization script (Wav2Lip)
    output_video_path = os.path.join(RESULTS_FOLDER, "deepfake_output.mp4")
    os.system(f"python ../../synchronization/sync_scripts/audio_video_sync.py {image_path} {audio_path} {output_video_path}")

    return send_file(output_video_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
