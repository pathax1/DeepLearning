# wav2lip.py
# Simplified implementation of Wav2Lip for synchronizing audio with generated face videos.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        # CNN encoder for audio features
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Video frame encoder (CNN)
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Combined feature decoder (lip-sync)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, mel_audio, video_frame):
        # Encode audio features
        audio_feat = self.audio_encoder(mel_audio)  # [batch, channels, seq_len]
        audio_feat = audio_feat.unsqueeze(-1)  # [batch, channels, seq_len, 1]

        # Encode video frames
        video_feat = self.video_encoder(video_frame)  # [batch, channels, height, width]

        # Combine audio and video features
        combined_feat = torch.cat((audio_feat, video_feat), dim=1)

        # Decode to generate synchronized frame
        synced_frame = self.decoder(combined_feat)

        return synced_frame

# Test quick forward pass
if __name__ == "__main__":
    mel_audio = torch.randn(2, 80, 16)  # Batch of mel-spectrogram audio features
    video_frame = torch.randn(2, 3, 64, 64)  # Batch of video frames

    model = Wav2Lip()
    output_frame = model(mel_audio, video_frame)
    print("Output Video Frame shape:", output_frame.shape)
