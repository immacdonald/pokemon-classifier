from torchvision.datasets import ImageFolder


class ImageFolderWithFilenames(ImageFolder):
    def __getitem__(self, index):
        """
        Overrides the default __getitem__ method to include filenames.

        Returns:
            image (Tensor): Transformed image.
            label (int): Label index.
            filename (str): Filename of the image.
        """
        path, label = self.samples[index]  # Get the file path and label
        image = self.loader(path)  # Load the image
        if self.transform is not None:
            image = self.transform(image)  # Apply transforms

        filename = path.split("/")[-1]  # Extract the filename
        return image, label, filename
