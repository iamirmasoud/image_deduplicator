import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from configs import images_dir, batch_size, embedding_size, image_resize
from images_dataset import ImagesDataset
from model import EncoderCNN

# Define a transform to pre-process the training images.
transformer = transforms.Compose(
    [
        transforms.Resize((image_resize, image_resize)),  # smaller edge of image resized to 256
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ImagesDataset(directory=images_dir, transform=transformer)
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False
)

# Initializing the encoder
encoder = EncoderCNN(embedding_size=embedding_size)
encoder.to(device)
encoder.eval()

full_features_df = pd.DataFrame([])
for images_batch, paths in data_loader:
    images = images_batch.to(device)
    encoder.zero_grad()

    # Passing the inputs through the CNN model
    with torch.no_grad():
        features = encoder(images)
    batch_df = pd.DataFrame(features.cpu().numpy(), index=paths)
    full_features_df = full_features_df.append(batch_df)

full_features_df.to_pickle("features.pkl")
