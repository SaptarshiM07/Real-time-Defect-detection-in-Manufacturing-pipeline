import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from model_training.utils.dataloader import get_split_loaders
from model_training.preprocessing.transforms import get_basic_transforms
from model_training.preprocessing.augmentation import get_train_augmentations
from train import train_model
from model_training.models.model_factory import get_model_by_name
import model_training.config as config

#models_to_run = ["resnet18", "mobilenet", "efficientnet", "baselinecnn"]
models_to_run = ["resnet18",  "baselinecnn"]
learning_rates = [0.1]

train_tf = get_train_augmentations()
val_tf = get_basic_transforms()

train_loader, val_loader = get_split_loaders(
    data_dir=config.TRAIN_DIR,
    train_transform=train_tf,
    val_transform=val_tf,
    batch_size=config.BATCH_SIZE,
    split_ratio=0.8
)

for model_name in models_to_run:
    for lr in learning_rates:
        with mlflow.start_run(run_name=f"{model_name}_lr{lr}"):
            # mlflow.log_param("model", model_name)
            # mlflow.log_param("lr", lr)
            # mlflow.log_param("batch_size", config.BATCH_SIZE)
            # mlflow.log_param("epochs", config.NUM_EPOCHS)

            model = get_model_by_name(model_name, num_classes=config.NUM_CLASSES)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=config.DEVICE,
                num_epochs=config.NUM_EPOCHS,
                model_name=model_name,
                save_dir=config.SAVE_DIR
            )