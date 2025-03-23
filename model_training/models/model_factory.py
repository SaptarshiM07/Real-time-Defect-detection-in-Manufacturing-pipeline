def get_model_by_name(model_name, num_classes=6, pretrained=True):
    """
    Returns the model instance given a model name.

    Args:
        model_name (str): One of ["resnet18", "mobilenet", "efficientnet", "baselinecnn"]
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights (for transfer learning models)

    Returns:
        model (torch.nn.Module): The initialized model
    """
    if model_name == "resnet18":
        from model_training.models.resnet18_model import get_resnet18
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)

    elif model_name == "mobilenet":
        from model_training.models.mobilenet_model import get_mobilenet
        return get_mobilenet(num_classes=num_classes, pretrained=pretrained)

    elif model_name == "efficientnet":
        from model_training.models.efficientnet_model import get_efficientnet
        return get_efficientnet(num_classes=num_classes, pretrained=pretrained)

    elif model_name == "baselinecnn":
        from model_training.models.baseline_cnn import BaselineCNN
        return BaselineCNN(num_classes=num_classes)

    else:
        raise ValueError(f"❌ Unknown model name: {model_name}")