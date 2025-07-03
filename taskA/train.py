# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os

from . import config
from . import model as model_def
from . import data_loader as dl
from . import utils

def train():
    """Main training and validation loop."""
    print("🚀 Starting Training Process...")
    print(f"Device: {config.DEVICE}")

    # --- DataLoaders ---
    train_loader, class_names = dl.create_dataloader(config.TRAIN_DIR, config.BATCH_SIZE, shuffle=True)
    val_loader, _ = dl.create_dataloader(config.VAL_DIR, config.BATCH_SIZE, shuffle=False)
    print(f"Found classes: {class_names}")

    # --- Model, Optimizer, Criterion ---
    model = model_def.get_model().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    
    print("\n--- Training Started ---")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation ---
        val_preds, val_labels = utils.evaluate(model, val_loader, config.DEVICE)
        val_acc = (torch.tensor(val_preds) == torch.tensor(val_labels)).sum().item() / len(val_labels)

        print(f"[Epoch {epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✨ New best model found! Saving to {config.MODEL_SAVE_PATH}")
            utils.save_model(model, config.MODEL_SAVE_PATH)
            
    print("\n--- Training Finished ---")
    print(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # --- Final Evaluation on Validation Set with Best Model ---
    print("\n--- Loading best model for final validation report ---")
    best_model = model_def.get_model(pretrained=False).to(config.DEVICE)
    best_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    final_preds, final_labels = utils.evaluate(best_model, val_loader, config.DEVICE)
    print("\n📋 Final Validation Metrics (from best model):")
    utils.print_metrics(final_labels, final_preds, target_names=class_names)

if __name__ == '__main__':
    train()