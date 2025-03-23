import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data/NEU-DET/train/images")
TEST_DIR = os.path.join(PROJECT_ROOT, "data/NEU-DET/validation/images") #for kafka producer real time

SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs/models")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs/logs")
PLOT_DIR = os.path.join(PROJECT_ROOT, "outputs/plots")

# Training params
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.01

# Model info
NUM_CLASSES = 6

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")