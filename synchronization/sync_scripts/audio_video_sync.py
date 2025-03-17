# audio_video_sync.py
# This script synchronizes generated voice audio with generated face video frames using the Wav2Lip model.

import os
import torch
import cv2
import numpy as np
from wav2lip.wav2lip import Wav2Lip
import librosa
import soundfile as sf

class AudioVideoSync:
    def __init__(self, wav2lip_model_path, audio_path, video_frames_folder, output_folder):
        self.wav2lip_model = self.load_model(wav2lip_model_path)
        self.audio_path = audio_path
        self.video_frames_path = video_frames_folder
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def load_model(self, model_path):
        model = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        return model

    # Method to load mel-spectrogram from audio file
    def audio_to_mel(self, audio_path):
        import librosa
        wav, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel = librosa.power_to_db(mel, ref=np.max)
        return torch.FloatTensor(mel).unsqueeze(0)

    def synchronize(self):
        video_frames = sorted([
            f for f in os.listdir(self.video_frames_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        mel_audio = self.load_audio(self.audio_path)

        synced_frames = []

        for frame_name in video_frames:
            frame_path = os.path.join(self.video_frames_path, frame)
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (64, 64))
            frame_tensor = torch.FloatTensor(frame.transpose(2, 0, 1)).unsqueeze(0)

            # Run Wav2Lip model
            synced_frame = self.wav2lip_model(mel, frame_tensor)
            synced_frame = synced_frame.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            synced_frame = (synced_frame * 255).astype(np.uint8)

            synced_frame_path = os.path.join(self.output_folder, f'synced_{frame_file}')
            cv2.imwrite(synced_frame_path, synced_frame)

        print("Audio-Video synchronization completed successfully.")

if __name__ == "__main__":
    wav2lip_model_path = "../wav2lip/wav2lip_model.pth"
    audio_path = "../../voice_generation/generated_audio/sample.wav"
    video_frames_folder = "../../face_generation/generated_faces/"
    output_folder = "../synchronized_videos/"

    sync = AudioVideoSynchronizer(
        wav2lip_model_path, audio_path, video_frames_path, output_folder
    )
    sync.synchronize()
