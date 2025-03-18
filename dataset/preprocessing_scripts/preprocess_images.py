# preprocess_images.py
# Preprocessing script for FFHQ image dataset

import os
import cv2
from tqdm import tqdm

class FacePreprocessor:
    def __init__(self, input_folder, output_folder, image_size=(256, 256)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_size = image_size

    # Method to preprocess a single image
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to load {image_path}")
            return None
        img = cv2.resize(img, self.image_size)
        return img

    # Method to preprocess all images in the dataset
    def process_dataset(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        images_list = os.listdir(self.input_folder)
        for filename in tqdm(images_list, desc="Processing Images"):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)

            preprocessed_img = self.preprocess_image(input_path)
            if preprocessed_img is not None:
                cv2.imwrite(output_path, preprocessed_img)

if __name__ == "__main__":
    input_folder = r"C:\Users\Autom\PycharmProjects\DeepLearning\dataset\FFHQ\original_images"
    output_folder = r"C:\Users\Autom\PycharmProjects\DeepLearning\dataset\FFHQ\preprocessed_images"

    processor = FacePreprocessor(input_folder, output_folder)
    processor.process_dataset()
