# this script downloads dataset from kagglehub
#
# if you dont wanna use kagglehub here is the link - download however you like
# https://www.kaggle.com/datasets/mdnurnabirana/english-handwritten-characters-and-digit
#
# place it inside dataset/ such that it looks like this:
# dataset/
#     Img/
#         img.png
#     english.csv
#
# in case dataset is deleted here is the data structure
#
#  Images:
#
#     Located in the Img/ folder
#     Each image is a handwritten character saved in PNG format
#     File naming pattern: imgXXX-YYY.png
#
# Labels:
#
#     Provided in english.csv
#     Two columns:
#         image: Relative file path to each image
#         label: Integer-encoded label (0–61) for the corresponding character class (A–Z, a–z, 0–9)

import os
import shutil

import kagglehub

path = kagglehub.dataset_download(
    "mdnurnabirana/english-handwritten-characters-and-digit"
)

dest = os.path.join(os.path.dirname(__file__), "dataset")
if not os.path.exists(dest):
    shutil.copytree(path, dest)
    print("Copied to:", dest)
else:
    print("Already exists:", dest)

shutil.rmtree(path)
