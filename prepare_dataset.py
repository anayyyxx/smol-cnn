# this script precoesses the data - you ask why? data too good - very big - small is okay - making small - who needs color?

import csv
import os
import random

import cv2

from config import (
    dataset_base_path,
    dataset_output_path,
    label_csv_path,
    train_val_test_split,
)
from map import char_to_int

# read image paths and labels from csv
csv_data = csv.reader(open(label_csv_path, "r"))
image_paths = []
image_labels = {}
for index, row in enumerate(csv_data):
    if index == 0:
        continue
    image_paths.append(row[0])
    image_labels[os.path.basename(row[0])] = char_to_int[row[1]]

# randomly split into train val and test
random.shuffle(image_paths)
train_size = int(len(image_paths) * train_val_test_split[0])
val_size = int(len(image_paths) * train_val_test_split[1])
train_paths = image_paths[:train_size]
val_paths = image_paths[train_size : train_size + val_size]
test_paths = image_paths[train_size + val_size :]

# make output directories
train_output_path = os.path.join(dataset_output_path, "train")
val_output_path = os.path.join(dataset_output_path, "val")
test_output_path = os.path.join(dataset_output_path, "test")
os.makedirs(train_output_path, exist_ok=True)
os.makedirs(val_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)

output_path_map = {
    "train": train_output_path,
    "val": val_output_path,
    "test": test_output_path,
}
img_path_map = {"train": train_paths, "val": val_paths, "test": test_paths}

# adding a progress bar to curb my anxiety
total_images = len(image_paths)
current_progress = 0
def print_and_update_progress():
    global current_progress
    print(f"\r processing image {current_progress}/{total_images}", end="")
    current_progress += 1


# pre process images
for split in ["train", "val", "test"]:
    output_path = output_path_map[split]
    img_paths = img_path_map[split]
    for img_path in img_paths:
        print_and_update_progress()
        img_path = os.path.join(dataset_base_path, img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_grayscale, (32, 32))
        img_name = os.path.basename(img_path)
        img_output_path = os.path.join(output_path, img_name)
        cv2.imwrite(img_output_path, img_resized)

# write labels csv with integer labels
label_output = os.path.join(dataset_output_path, "labels.csv")
with open(label_output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])
    for img_name, label in image_labels.items():
        writer.writerow([img_name, label])
