import torch
from tqdm import tqdm
import os
import mlflow
import mlflow.pytorch
from evaluate import evaluate_model
from model_training.utils.metrics import compute_metrics, save_metrics_to_csv, plot_metrics_from_csv

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, save_dir, patience=5):
    model.to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0

    # Assume mlflow.start_run() is handled by the parent script
    mlflow.log_param("model", model_name)
    mlflow.log_param("lr", optimizer.param_groups[0]['lr'])
    mlflow.log_param("batch_size", train_loader.batch_size)
    mlflow.log_param("epochs", num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        preds, labels = evaluate_model(model, val_loader, device)
        acc, precision, recall, f1 = compute_metrics(preds, labels)

        print(f"[Epoch {epoch+1}] Val - Acc: {acc:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")

        mlflow.log_metric("accuracy", acc, step=epoch+1)
        mlflow.log_metric("precision", precision, step=epoch+1)
        mlflow.log_metric("recall", recall, step=epoch+1)
        mlflow.log_metric("f1_score", f1, step=epoch+1)

        save_path = os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model: {save_path}")

        # Early stopping check
        if acc > best_val_acc:
            best_val_acc = acc
            epochs_no_improve = 0
            mlflow.pytorch.log_model(model, artifact_path=f"{model_name}_lr{optimizer.param_groups[0]['lr']}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break