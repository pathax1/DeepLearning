# train_face_gan.py
# Script to train the face generation GAN using StyleGAN2 architecture.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os
from PIL import Image
import glob

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from face_generation.models.stylegan2 import Generator, Discriminator


# Custom dataset class to load images from a flat directory
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # ✅ Support both .jpg and .png files
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label (not used)


class FaceGANTrainer:
    def __init__(self, data_path, output_path, latent_dim=100, batch_size=64, epochs=50, lr=0.0002):
        self.data_path = data_path
        self.output_path = output_path
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
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
        dataset = ImageDataset(image_dir=self.data_path, transform=transform)  # Using the custom dataset class

        # ✅ Debugging: Print number of images found
        num_images = len(dataset)
        print(f"✅ Found {num_images} images in {self.data_path}")

        if num_images == 0:
            raise ValueError(f"❌ No images found in {self.data_path}. Check your dataset!")

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    # GAN training method
    def train(self):
        os.makedirs(self.output_path, exist_ok=True)  # Ensure output directory exists

        for epoch in range(self.epochs):
            for images, _ in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_size = images.size(0)

                real_images = images.to(self.device)
                real_labels = torch.ones(batch_size).to(self.device)  # FIXED shape
                fake_labels = torch.zeros(batch_size).to(self.device)  # FIXED shape

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.optimizer_d.zero_grad()

                # Real images
                outputs_real = self.discriminator(real_images).view(-1)
                loss_real = self.criterion(outputs_real, real_labels)

                # Fake images
                noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
                fake_images = self.generator(noise)
                outputs_fake = self.discriminator(fake_images.detach()).view(-1)
                loss_fake = self.criterion(outputs_fake, fake_labels)

                loss_disc = loss_real + loss_fake
                loss_disc.backward()
                self.optimizer_d.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.optimizer_g.zero_grad()
                outputs_fake = self.discriminator(fake_images).view(-1)
                loss_gen = self.criterion(outputs_fake, real_labels)
                loss_gen.backward()
                self.optimizer_g.step()

            print(f"Epoch [{epoch+1}/{self.epochs}] - Discriminator Loss: {loss_disc.item():.4f}, Generator Loss: {loss_gen.item():.4f}")

            # Save generator checkpoints periodically
            torch.save(self.generator.state_dict(), os.path.join(self.output_path, f'generator_epoch_{epoch+1}.pth'))

        print("✅ Training completed successfully!")

# Main execution block
if __name__ == "__main__":
    trainer = FaceGANTrainer(
        data_path=r"C:\Users\Autom\PycharmProjects\DeepLearning\dataset\FFHQ\preprocessed_images\all_faces",  # Fixed dataset path
        output_path=r"C:\Users\Autom\PycharmProjects\DeepLearning\face_generation\generated_faces",  # Fixed output path
        latent_dim=100,
        batch_size=64,
        epochs=50
    )
    trainer.train()
