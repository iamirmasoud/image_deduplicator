import logging
import os
import sys

import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs import (batch_size, delete_files, embedding_size, epsilon,
                     extensions, image_resize, images_dir)
from images_dataset import ImagesDataset
from model import EncoderCNN

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if len(sys.argv) > 1:
    images_dir = sys.argv[1]

if not os.path.isdir(images_dir):
    logger.error(f"Directory '{images_dir}' does not exist.")
    raise NotADirectoryError

# Define a transform to pre-process the training images.
transformer = transforms.Compose(
    [
        transforms.Resize((image_resize, image_resize)),
        # convert the PIL Image to a tensor
        transforms.ToTensor(),
        # normalize image for pre-trained model
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {str(device).upper()}")

dataset = ImagesDataset(
    directory=images_dir, extensions=extensions, transform=transformer
)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
logger.info(f"Found {len(dataset)} image files in directory '{images_dir}'.")

# Initializing the encoder
encoder = EncoderCNN(embedding_size=embedding_size)
encoder.to(device)
encoder.eval()

# ------ Extracting features ------
features_df = pd.DataFrame([])
for images_batch, paths in tqdm(data_loader):
    images = images_batch.to(device)
    encoder.zero_grad()

    # Passing the inputs through the CNN model
    with torch.no_grad():
        features = encoder(images)
    batch_df = pd.DataFrame(features.cpu().numpy(), index=paths)
    features_df = features_df.append(batch_df)

logging.info("Features have been extracted .")

# ------ Clustering ------
model = DBSCAN(min_samples=1, eps=epsilon)

clusters = pd.DataFrame(
    {"path": features_df.index, "label": model.fit_predict(features_df)}
).sort_values(["label", "path"], ascending=False)

# select one of elements from each cluster to keep
list_to_keep = set(clusters.groupby("label")["path"].first())
all_files = set(clusters["path"])

files_to_remove = all_files - list_to_keep
logger.info(f"Found {len(files_to_remove)} duplicate files:\n {files_to_remove}")

# ------ Removing files ------
if delete_files:
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except OSError as e:
            print("Cannot delete file ''%s': %s" % (file_path, e.strerror))
    logger.info(f"Successfully deleted {len(files_to_remove)} duplicate files.")
