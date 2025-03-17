# evaluate_quality.py
# Evaluates the quality of generated faces and audio using standard metrics (FID, SSIM).

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
import torch.nn.functional as F
import cv2
from scipy import linalg
import librosa
from skimage.metrics import structural_similarity as ssim

class DeepfakeEvaluator:
    def __init__(self, real_images_path, generated_images_path):
        self.real_images_path = real_images_path
        self.generated_images_path = generated_images_path

    # Feature extraction using pretrained CNN (simple embedding)
    def extract_features(self, images_path):
        images = []
        for file in os.listdir(images_path):
            img = cv2.imread(os.path.join(images_path, file))
            img_resized = cv2.resize(img, (299, 299))
            images.append(img.astype(np.float32) / 255.)

        images = np.stack(images)
        images = images.transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(images)

        return images_tensor.view(images_tensor.size(0), -1).numpy()

    def calculate_fid(self):
        real_features = self.extract_features(self.real_images_path)
        generated_features = self.extract_features(self.generated_images_path)

        mu_real, sigma_real = real_images.mean(axis=0), np.cov(real_images, rowvar=False)
        mu_gen, sigma_gen = generated_images.mean(axis=0), np.cov(generated_images, rowvar=False)

        # Compute Frechet Distance
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        fid_score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean.real)

        return fid_score

    def evaluate_ssim(self, img_path_real, img_generated_path):
        img_real = cv2.imread(real_img)
        img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)

        img_gen = cv2.imread(generated_img_path)
        img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(img_real, img_gen, full=True)
        return score

    def evaluate_dataset(self):
        generated_images_files = os.listdir(self.generated_images_path)
        scores = []

        for img_file in generated_images_files:
            real_img_path = os.path.join(self.real_images_path, img_file)
            generated_img_path = os.path.join(self.generated_images_path, img_file)

            ssim_score = self.evaluate_ssim(real_img_path, generated_img_path)
            print(f"SSIM score for {img_file}: {score:.4f}")

        # Example calculation for FID
        fid_score = self.calculate_fid()
        print(f"FID Score (Dataset-level quality): {fid_score:.4f}")

if __name__ == "__main__":
    real_images_path = "../../dataset/FFHQ/preprocessed_images"
    generated_images_path = "../../face_generation/generated_faces"

    evaluator = DeepfakeEvaluator(real_images_path, generated_images_path)
    evaluator.evaluate_dataset()
