import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def rgba_to_rgb(image):
    if image.mode == 'RGBA':
        # Replace transparent areas with a white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return background
    else:
        return image

def calculate_mean_std(dataset_loader):
    # Accumulators for mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in dataset_loader:
        # Check shape and normalize properly
        assert images.ndim == 4, "Images must have shape (batch, 3, H, W)"
        assert images.size(1) == 3, "Images must have 3 channels (RGB)"
        
        batch_samples = images.size(0) # Batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)


    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()


def main():
    # Define transformations without normalization for calculation
    transforms_for_stats = transforms.Compose([
        transforms.Lambda(rgba_to_rgb),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    # Load dataset
    dir: str = os.path.dirname(p=os.path.realpath(filename=__file__))
    data_dir = os.path.join(dir, "dataset")
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms_for_stats)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Calculate mean and std
    mean, std = calculate_mean_std(dataset_loader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == '__main__':
    main()