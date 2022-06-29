import torch.nn as nn
import torchvision.models as models


# ----------- Encoder ------------
class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # disable learning for parameters
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embedding_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
