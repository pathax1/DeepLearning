# preprocess_audio.py
# This script preprocesses audio files for the Deepfake project by normalizing audio data and converting it to a standard format.

import os
import librosa
import soundfile as sf
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(self, input_folder, output_folder, target_sr=22050):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_sr = target_sr  # Standard sampling rate for audio

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    # Method to preprocess a single audio file
    def preprocess_audio_file(self, audio_path, output_path):
        # Load audio file and resample
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        # Normalize audio
        audio = librosa.util.normalize(audio)
        # Save preprocessed audio as WAV
        sf.write(output_path, audio, self.target_sr)

    # Method to preprocess entire dataset
    def preprocess_dataset(self):
        audio_files = [
            file for file in os.listdir(self.input_folder)
            if file.lower().endswith(('.mp3', '.wav'))
        ]

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for audio_file in tqdm(audio_files, desc="Processing Audio Files"):
            input_path = os.path.join(self.input_folder, audio_file)
            output_file = os.path.splitext(audio_file)[0] + ".wav"
            output_path = os.path.join(self.output_folder, output_file)
            self.preprocess_audio(input_path, output_path)


if __name__ == "__main__":
    input_folder = "../audio_samples/original_audio"
    output_folder = "../audio_samples/processed_audio"

    preprocessor = AudioPreprocessor(input_folder, output_folder)
    preprocessor.preprocess_dataset()
