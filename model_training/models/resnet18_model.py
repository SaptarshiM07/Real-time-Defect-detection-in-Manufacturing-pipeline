import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=6, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model