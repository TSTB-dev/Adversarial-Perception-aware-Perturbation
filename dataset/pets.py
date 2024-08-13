import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
import tarfile

from torchvision.datasets.utils import download_url

class PetsDataset(Dataset):
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    files = {
        "images": "images.tar.gz",
        "annotations": "annotations.tar.gz"
    }

    def __init__(self, root, transform=None, download=False, train=True):
        self.root = root
        self.transform = transform
        self.train = train

        if download:
            self.download()

        self.images, self.labels = self.load_data()

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        for key, filename in self.files.items():
            url = self.base_url + filename
            download_url(url, self.root, filename)
            self.extract_file(os.path.join(self.root, filename))

    def extract_file(self, file_path):
        if file_path.endswith("tar.gz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)
        elif file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

    def load_data(self):
        if self.train:
            annotations_path = os.path.join(self.root, "annotations", "trainval.txt")
        else:
            annotations_path = os.path.join(self.root, "annotations", "test.txt")
        
        images = []
        labels = []
        with open(annotations_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                image_name = parts[0]
                label = int(parts[1]) - 1
                images.append(os.path.join(self.root, "images", image_name + ".jpg"))
                labels.append(label)
                
        # set number of classes
        self.classes = sorted(set(labels))
        self.num_classes = len(set(labels))
        
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": str(label),
            "path": image_path
        }

if __name__ == "__main__":
    dataset = PetsDataset(root="data", download=True)
    print(len(dataset))
    print(dataset[0]["image"].size)