# train_face_gan.py
# Script to train the face generation GAN using StyleGAN2 architecture.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from models.stylegan2 import Generator, Discriminator

class FaceGANTrainer:
    def __init__(self, data_path, latent_dim=100, batch_size=64, epochs=50, lr=0.0002):
        self.data_path = data_path
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = 0.0002
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize generator and discriminator models
        self.generator = Generator(latent_dim=self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Data loading
        self.dataloader = self.load_data()

    # Method to load and transform data
    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = ImageFolder(root="../../dataset/FFHQ/preprocessed_images", transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        return dataloader

    # GAN training method
    def train(self):
        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            for images, _ in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_size = images.size(0)

                real_images = images.to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.discriminator.zero_grad()

                # Real images
                outputs_real = self.discriminator(real_images).view(-1)
                loss_real = criterion(outputs_real, real_labels)

                # Fake images
                noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
                fake_images = self.generator(noise)
                outputs_fake = self.discriminator(fake_images.detach()).view(-1)
                loss_fake = criterion(outputs_fake, fake_labels)

                loss_disc = loss_real + loss_fake
                self.discriminator.zero_grad()
                loss_disc.backward()
                self.optimizerD.step()

                # Train Generator
                self.generator.zero_grad()
                outputs_fake = self.discriminator(fake_images).view(-1)
                loss_gen = criterion(outputs_fake, real_labels)
                loss_gen.backward()
                self.optimizer_g.step()

            print(f"Epoch [{epoch+1}/{self.epochs}] - Discriminator Loss: {loss_disc.item():.4f}, Generator Loss: {loss_gen.item():.4f}")

            # Save generator checkpoints periodically
            os.makedirs('../generated_faces/', exist_ok=True)
            torch.save(self.generator.state_dict(), f'../generated_faces/generator_epoch_{epoch+1}.pth')

        print("Training completed successfully!")

# Main execution block
if __name__ == "__main__":
    trainer = FaceGANTrainer(
        data_folder="../../dataset/FFHQ/preprocessed_images",
        latent_dim=100,
        batch_size=64,
        epochs=50
    )
    trainer.train()
