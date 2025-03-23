import torch
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def compute_metrics(preds, labels):
    """
    Compute accuracy, precision, recall, and F1 score.
    Expects:
    - preds: list of predicted tensors
    - labels: list of ground truth tensors
    """
    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    acc = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, average='macro', zero_division=0) * 100
    recall = recall_score(labels, preds, average='macro', zero_division=0) * 100
    f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100

    return acc, precision, recall, f1


def save_metrics_to_csv(epoch, acc, precision, recall, f1, model_name, save_dir):
    """
    Save metrics to a CSV file for each model.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_metrics.csv")

    new_row = pd.DataFrame({
        'epoch': [epoch],
        'accuracy': [acc],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(file_path, index=False)
    return file_path


def plot_metrics_from_csv(file_path, save_plot_dir):
    """
    Plot accuracy, precision, recall, F1 over epochs from a saved CSV file.
    """
    df = pd.read_csv(file_path)
    model_name = os.path.basename(file_path).split("_metrics")[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['accuracy'], label='Accuracy')
    plt.plot(df['epoch'], df['precision'], label='Precision')
    plt.plot(df['epoch'], df['recall'], label='Recall')
    plt.plot(df['epoch'], df['f1_score'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score (%)')
    plt.title(f'{model_name} - Metrics Over Epochs')
    plt.legend()
    plt.grid(True)

    os.makedirs(save_plot_dir, exist_ok=True)
    save_path = os.path.join(save_plot_dir, f"{model_name}_metrics.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()