from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch


class ImageNetSketchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        # Ensure the image is in RGB format
        # Some images in ImageNet-Sketch might be in 'L' (grayscale) mode
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        elif not isinstance(image, Image.Image):
            # If it's a numpy array or other format, convert to PIL first
            image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label
