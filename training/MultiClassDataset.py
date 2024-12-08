import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiClassDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset to handle flat folder structure with class encoding in filenames.
        Args:
            root (str): Path to the dataset directory containing images.
            transform (callable, optional): Transform to apply to images.
        """
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, file) for file in os.listdir(root) if file.lower().endswith(("png", "jpg", "jpeg"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image tensor.
            label (Tensor): One-hot encoded label tensor.
            filename (str): Plain text of the file name.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Extract one-hot encoded label from filename
        filename = os.path.basename(image_path)
        label = parse_labels_from_filename(filename)

        return image, label, filename


def parse_labels_from_filename(filename):
    """
    Parse one-hot encoded labels from the filename.
    Example: 'bulbasaur-00010000000100000-1.png' -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    """
    encoding = filename.split("-")[1]
    return torch.tensor([int(bit) for bit in encoding], dtype=torch.float32)
