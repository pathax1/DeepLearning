# tacotron2.py
# Tacotron2 Model for Text-to-Speech Voice Generation using PyTorch
import torch
import torch.nn as nn

class Tacotron2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, mel_channels=80):
        super(Tacotron2, self).__init__()

        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Encoder: Converts text to hidden states
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Decoder: Generates Mel spectrogram frames
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)

        # Linear projection to Mel spectrogram
        self.mel_linear = nn.Linear(hidden_dim, mel_channels)

        # Post-processing ConvNet (simplified)
        self.postnet = nn.Sequential(
            nn.Conv1d(mel_channels, mel_channels, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(mel_channels, mel_channels, kernel_size=5, padding=2)
        )

    def forward(self, text_inputs):
        # Text embeddings
        embeddings = self.embedding(text_inputs)

        # Encoder
        encoder_outputs, _ = self.encoder(embeddings)

        # Decoder (uses encoder outputs)
        decoder_outputs, _ = self.decoder(encoder_outputs)

        # Mel spectrogram prediction
        mel_outputs = self.mel_linear(decoder_outputs)

        # Post-processing step
        mel_outputs_post = mel_outputs + self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)

        return mel_outputs, mel_outputs_post

# Quick model testing
if __name__ == "__main__":
    vocab_size = 50  # Define based on your character set
    batch_size = 2
    seq_length = 20

    model = Tacotron2(vocab_size=vocab_size)
    dummy_text = torch.randint(0, vocab_size, (batch_size, seq_length))

    mel_out, mel_post = model(dummy_text)
    print("Mel Output shape:", mel_out.shape)
    print("Postnet Output shape:", mel_post.shape)
