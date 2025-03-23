import torch.nn as nn
from torchvision import models

def get_mobilenet(num_classes=6, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model