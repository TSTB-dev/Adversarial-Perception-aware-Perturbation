import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
import tarfile
import h5py
import numpy as np

from torchvision.datasets.utils import download_url

import torchvision
from tqdm import tqdm

LABEL_TO_CLASSNAME = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck",
}

class STL10Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True, with_unlabeled=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.with_unlabeled = with_unlabeled

        if download:
            self.download()

        if self.train:
            if self.with_unlabeled:
                split = "train+unlabeled"
            else:
                split = "train"
        else:
            split = "test"
        self.dataset = torchvision.datasets.STL10(root=self.root, split=split, transform=self.transform, download=False)

        print(f"Number of samples in {split} set: {len(self.dataset)}")

        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

    def download(self):
        if os.path.exists(os.path.join(self.root, "stl10_binary")):
            print("Dataset already downloaded")
            return
        # Download the dataset using torchvision's download functionality
        torchvision.datasets.STL10(root=self.root, split='train', download=True)
        # One of {‘train’, ‘test’, ‘unlabeled’, ‘train+unlabeled’}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": LABEL_TO_CLASSNAME[label],
            "path": None  # Since STL10 data doesn't have paths, we can set it to None or str(index)
        }


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # STL10 images are 96x96
        transforms.ToTensor(),
    ])

    dataset = STL10Dataset(root="data", transform=transform, download=True, train=True)
    print(len(dataset))
    print(dataset[0]["image"].size())
    print(dataset[0]["caption"])
    print(dataset[0]["label"])
    print(dataset[0]["path"])
    
    # save image to file
    img = dataset[0]["image"]
    torchvision.utils.save_image(img, "stl10_example.png")
    
