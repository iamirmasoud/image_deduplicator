import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs import (batch_size, embedding_size, extensions, image_resize,
                     images_dir)
from images_dataset import ImagesDataset
from model import EncoderCNN

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

dataset = ImagesDataset(
    directory=images_dir, extensions=extensions, transform=transformer
)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
logger.info(f"Found {len(dataset)} files in the directory.")
# Initializing the encoder
encoder = EncoderCNN(embedding_size=embedding_size)
encoder.to(device)
encoder.eval()

full_features_df = pd.DataFrame([])
for images_batch, paths in tqdm(data_loader):
    images = images_batch.to(device)
    encoder.zero_grad()

    # Passing the inputs through the CNN model
    with torch.no_grad():
        features = encoder(images)
    batch_df = pd.DataFrame(features.cpu().numpy(), index=paths)
    full_features_df = full_features_df.append(batch_df)

full_features_df.to_pickle("features.pkl")
logging.info("Features have been extracted and stored to 'features.pkl'.")
