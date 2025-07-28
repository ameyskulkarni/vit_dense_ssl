import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class ContrastiveImageDataset(Dataset):
    def __init__(self, root, transform1=None, transform2=None):
        """
        Dataset that returns two differently augmented versions of the same image.

        Args:
            root: Path to the image directory
            transform1: First set of transforms
            transform2: Second set of transforms (if None, uses transform1)
        """
        self.dataset = ImageFolder(root)
        self.transform1 = transform1
        self.transform2 = transform2 if transform2 is not None else transform1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Apply two different augmentations to the same image
        image1 = self.transform1(image) if self.transform1 else image
        image2 = self.transform2(image) if self.transform2 else image

        return (image1, image2), label