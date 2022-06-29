import pathlib

from PIL import Image
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def list_files(
        self,
    ):
        """
        List all files in a directory with the given extensions.
        """
        files = [
            str(file)
            for ext in self.extensions
            for file in pathlib.Path(self.directory).rglob(f"*.{ext}")
        ]
        return files

    def __init__(
        self,
        directory,
        extensions=("jpg", "jpeg", "png", "bmp"),
        transform=None,
    ):
        self.directory = directory
        self.extensions = extensions
        self.transform = transform
        self.image_paths = self.list_files()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.image_paths)
