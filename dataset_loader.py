import csv
import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from config import dataset_base_path, processed_label_csv_path


class SmolCNNDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.split = split
        self.image_paths = os.listdir(f"dataset/{split}")
        self.image_paths = [
            img_name for img_name in self.image_paths if img_name.endswith(".png")
        ]
        self.csv_data = csv.reader(open(processed_label_csv_path, "r"))
        self.img_label_map = {}
        for row in self.csv_data:
            self.img_label_map[row[0]] = row[1]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(
            dataset_base_path, self.split, self.image_paths[index]
        )
        img_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img_grayscale).float().unsqueeze(0)
        img_tensor_normalised = img_tensor / 255.0
        label = int(self.img_label_map[self.image_paths[index]])
        return img_tensor_normalised, label


def get_dataloader(split, batch_size):
    dataset = SmolCNNDataset(split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train")
    )  # only shuffle for training
    return dataloader
