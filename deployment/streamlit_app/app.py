# app.py
# Streamlit web application for interactive Deepfake Voice & Face Generation

import streamlit as st
import os
from synchronization.sync_scripts.audio_video_sync import DeepfakeSynchronizer

# Page Configuration
st.set_page_config(page_title="Deepfake Generator", layout="centered")

# Title
st.title("🧑🏻 Deepfake Voice & Face Generation App")

# Upload Image and Audio
image_file = st.file_uploader("Upload Face Image (JPEG/PNG):", type=["png", "jpg", "jpeg"])
audio_file = st.file_uploader("Upload Audio (WAV format):", type=["wav"])

if st.button("Generate Deepfake"):
    if image_file and audio_file:
        # Save uploaded files temporarily
        os.makedirs("temp_uploads", exist_ok=True)
        image_path = os.path.join("temp_uploads", image_file.name)
        audio_path = os.path.join("temp_uploads", audio_file.name)

        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        st.success("✅ Files Uploaded Successfully!")

        # Generate synchronized video
        synchronizer = DeepfakeSynchronizer(image_path, audio_path, "temp_outputs")
        output_video_path = synchronizer.synchronize()

        st.video(output_video_path)

        # Cleanup temporary files after processing
        os.remove(image_path)
        os.remove(audio_path)

    else:
        st.warning("Please upload both an image and audio file to proceed.")
