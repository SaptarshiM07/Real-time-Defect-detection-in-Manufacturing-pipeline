from torchvision import transforms

def get_basic_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels =3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])