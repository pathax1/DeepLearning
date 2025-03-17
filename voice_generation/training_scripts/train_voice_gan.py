# train_voice_gan.py
# Training script for Tacotron2 and WaveGlow models for Voice Generation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import os
from tqdm import tqdm
from models.tacotron2 import Tacotron2
from models.waveglow import WaveGlow

# Custom Dataset Class
class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_folder, sample_rate=22050):
        self.audio_folder = audio_folder
        self.audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_folder, self.audio_files[idx])
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=80)
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)
        return mel_tensor

class VoiceGANTrainer:
    def __init__(self, data_path, epochs=50, batch_size=16, learning_rate=0.0001):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.tacotron2 = Tacotron2(vocab_size=50).to(self.device)
        self.waveglow = WaveGlow().to(self.device)

        # Optimizers
        self.optimizer_tacotron = torch.optim.Adam(self.tacotron2.parameters(), lr=self.lr)
        self.optimizer_waveglow = torch.optim.Adam(self.waveglow.parameters(), lr=self.lr)

        # Loss
        self.criterion = nn.MSELoss()

        # Dataset
        self.dataset = VoiceDataset(self.data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for mel_spectrograms in self.dataloader:
                mel_spectrograms = mel_spectrograms.to(self.device)

                # Tacotron2 forward pass (in practice, text embeddings would be used)
                predicted_mels = self.tacotron2(mel_spectrograms)

                # WaveGlow forward pass
                generated_audio = self.waveglow(mel_spectrograms)

                # Calculate simple MSE loss
                loss = nn.MSELoss()(generated_audio, mel_spectrograms)

                # Backpropagation and optimization
                self.optimizer_tacotron.zero_grad()
                self.optimizer_waveglow.zero_grad()
                loss.backward()
                self.optimizer_tacotron.step()
                self.optimizer_waveglow.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

            # Saving checkpoints periodically
            torch.save(self.tacotron2.state_dict(), f'../models/tacotron2_epoch_{epoch}.pth')
            torch.save(self.waveglow.state_dict(), f'../models/waveglow_epoch_{epoch}.pth')

            print(f"Epoch [{epoch+1}/{self.epochs}], Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    trainer = VoiceGANTrainer(
        data_path="../../dataset/audio_samples/processed_audio",
        epochs=50,
        batch_size=16,
        learning_rate=0.0001
    )
    trainer.train()
