# src/utils.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

def save_model(model, path):
    """Saves the model's state dictionary."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to: {path}")

def evaluate(model, dataloader, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader for the evaluation data.
        device (torch.device): The device to run evaluation on (e.g., 'cuda').

    Returns:
        tuple: A tuple containing lists of all predictions and all true labels.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_preds, all_labels

def print_metrics(labels, preds, target_names):
    """
    Calculates and prints key classification metrics.

    Args:
        labels (list or np.array): The true labels.
        preds (list or np.array): The model's predictions.
        target_names (list): The names of the classes for the report.
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    recall = recall_score(labels, preds, average='binary', zero_division=0)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    
    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, target_names=target_names, zero_division=0))
    
    return accuracy, precision, recall, f1