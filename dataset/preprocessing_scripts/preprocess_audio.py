# preprocess_audio.py
# Recursively processes all .flac files in subfolders, converts to .wav, and saves in a structured output directory.

import os
import librosa
import soundfile as sf
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self, input_folder, output_folder, target_sr=22050):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_sr = target_sr  # Standard sampling rate for audio

        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_audio(self, audio_path, output_path):
        """ Loads audio, normalizes it, and saves as a .wav file. """
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        audio = librosa.util.normalize(audio)
        sf.write(output_path, audio, self.target_sr)

    def preprocess_dataset(self):
        """ Recursively finds all .flac files and converts them to .wav while preserving subfolder structure. """
        audio_files = []

        print(f"Scanning directory: {self.input_folder}")

        # Walk through all subdirectories to collect .flac files
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if file.lower().endswith('.flac'):
                    full_path = os.path.join(root, file)
                    audio_files.append(full_path)

        if len(audio_files) == 0:
            print("⚠️ No .flac files found! Check if dataset is correctly placed.")
            return

        print(f"Found {len(audio_files)} .flac files. Starting processing...")

        # Process all collected audio files
        for audio_path in tqdm(audio_files, desc="Processing Audio Files"):
            # Preserve subfolder structure in output folder
            relative_path = os.path.relpath(audio_path, self.input_folder)
            output_path = os.path.join(self.output_folder, relative_path).replace(".flac", ".wav")

            # Create corresponding subdirectories in output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Process and save the audio file
            self.preprocess_audio(audio_path, output_path)

        print("✅ Audio Preprocessing Completed Successfully!")

if __name__ == "__main__":
    input_folder = r"C:\Users\Autom\PycharmProjects\DeepLearning\dataset\audio_samples\original_audio\LibriSpeech\train-clean-360"
    output_folder = r"C:\Users\Autom\PycharmProjects\DeepLearning\dataset\audio_samples\processed_audio"

    processor = AudioPreprocessor(input_folder, output_folder)
    processor.preprocess_dataset()
