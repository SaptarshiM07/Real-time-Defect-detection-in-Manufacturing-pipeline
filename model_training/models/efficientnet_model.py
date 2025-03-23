import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes=6, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model