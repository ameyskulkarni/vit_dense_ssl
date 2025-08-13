import torch
import json
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple, Any


class ImageNetV2Dataset(Dataset):
    """
    Custom dataset for ImageNet-V2 that handles class mapping online
    """

    def __init__(self, root: str, transform=None):
        """
        Args:
            root: Path to ImageNet-V2 root directory (contains folders 0, 1, 2, ...)
            transform: Transform to apply to images
        """
        self.root = root
        self.transform = transform

        # Use ImageFolder to handle the folder structure and image loading
        self.image_folder = ImageFolder(root=root, transform=None)  # We'll apply transform manually
        self.imagenet_v2_class_idx = self.image_folder.class_to_idx # class name: idx
        self.imagenet_v2_idx_class = {v: k for k, v in self.imagenet_v2_class_idx.items()}

        print(f"Loaded {len(self.image_folder)} images from ImageNet-V2")
        print(f"Number of classes: {len(self.image_folder.classes)}")

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get image and ImageNet-V2 class index from ImageFolder
        image, v2_class_idx = self.image_folder[idx]


        # Map V2 class index to V1 class index
        v1_class_idx = int(self.imagenet_v2_idx_class[v2_class_idx])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, v1_class_idx


class ImageNetADataset(Dataset):
    """
    Custom dataset for ImageNet-A that handles class mapping online
    """

    def __init__(self, root: str, mapping: Dict[str, int], transform=None):
        """
        Args:
            root: Path to ImageNet-A root directory (contains folders 0, 1, 2, ...)
            mapping: Mapping dictionary where the key is the imagenet v1 class str and value is the label used
            for this key when training and evaluating.
            transform: Transform to apply to images
        """
        self.root = root
        self.transform = transform
        self.mapping_imagenet_v1 = mapping

        # Use ImageFolder to handle the folder structure and image loading
        self.image_folder = ImageFolder(root=root, transform=None)  # We'll apply transform manually
        self.imagenet_A_class_idx = self.image_folder.class_to_idx # class_name: idx
        self.imagenet_A_idx_class = {v: k for k, v in self.imagenet_A_class_idx.items()} # idx: class_name

        print(f"Loaded {len(self.image_folder)} images from ImageNet-A")
        print(f"Number of classes: {len(self.image_folder.classes)}")

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get image and ImageNet-V2 class index from ImageFolder
        image, A_class_idx = self.image_folder[idx]


        # Map A class index to V1 class index
        v1_class_idx = int(self.mapping_imagenet_v1[self.imagenet_A_idx_class[A_class_idx]])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, v1_class_idx


class ImageNetRDataset(Dataset):
    """
    Custom dataset for ImageNet-R that handles class mapping online
    """

    def __init__(self, root: str, mapping: Dict[str, int], transform=None):
        """
        Args:
            root: Path to ImageNet-R root directory (contains folders 0, 1, 2, ...)
            mapping: Mapping dictionary where the key is the imagenet v1 class str and value is the label used
            for this key when training and evaluating.
            transform: Transform to apply to images
        """
        self.root = root
        self.transform = transform
        self.mapping_imagenet_v1 = mapping

        # Use ImageFolder to handle the folder structure and image loading
        self.image_folder = ImageFolder(root=root, transform=None)  # We'll apply transform manually
        self.imagenet_R_class_idx = self.image_folder.class_to_idx # class_name: idx
        self.imagenet_R_idx_class = {v: k for k, v in self.imagenet_R_class_idx.items()} # idx: class_name

        print(f"Loaded {len(self.image_folder)} images from ImageNet-R")
        print(f"Number of classes: {len(self.image_folder.classes)}")

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get image and ImageNet-V2 class index from ImageFolder
        image, R_class_idx = self.image_folder[idx]


        # Map A class index to V1 class index
        v1_class_idx = int(self.mapping_imagenet_v1[self.imagenet_R_idx_class[R_class_idx]])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, v1_class_idx
