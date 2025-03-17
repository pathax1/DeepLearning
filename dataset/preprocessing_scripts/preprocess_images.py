# preprocess_images.py
# Preprocessing script for FFHQ image dataset
import os
import cv2
from tqdm import tqdm


class FacePreprocessor:
    def __init__(self, input_dir, output_folder, image_size=(256, 256)):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_size = image_size

    # method to preprocess images
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_size)
        return img

    # method to preprocess all images in dataset
    def process_dataset(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        images_list = os.listdir(self.input_folder)
        for filename in tqdm(images_files):
            input_path = os.path.join(self.input_folder, filename)
            preprocessed_img = self.preprocess_image(input_folder + '/' + filename)
            output_path = self.output_folder + '/' + filename
            cv2.imwrite(output_path, preprocessed_img)


if __name__ == "__main__":
    import os
    import cv2
    from tqdm import tqdm

    input_folder = "../FFHQ/original_images"
    output_folder = "../FFHQ/preprocessed_images"

    processor = ImagePreprocessor(input_folder, output_folder)
    processor.preprocess_dataset()
