# src/test.py
import torch
import argparse
import os

from . import config
from . import model as model_def
from . import data_loader as dl
from . import utils

def test(data_path, model_path):
    """
    Tests a trained model on a given dataset.

    Args:
        data_path (str): The path to the test data directory.
        model_path (str): The path to the saved model state dictionary.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data path not found at '{data_path}'")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'")
        return

    print("🧪 Starting Testing Process...")
    print(f"Device: {config.DEVICE}")
    print(f"Loading model from: {model_path}")
    print(f"Testing on data from: {data_path}")

    # --- Model ---
    # Load architecture with pretrained=False, as we are loading our own weights
    model = model_def.get_model(pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))

    # --- DataLoader ---
    # The folder structure of test data should be:
    # data_path/
    #   ├── class_a/
    #   │   └── ...
    #   └── class_b/
    #       └── ...
    test_loader, class_names = dl.create_dataloader(data_path, config.BATCH_SIZE, shuffle=False)
    print(f"Found test classes: {class_names}")

    # --- Evaluation ---
    test_preds, test_labels = utils.evaluate(model, test_loader, config.DEVICE)
    
    # --- Print Results ---
    print("\n--- Test Set Evaluation Results ---")
    utils.print_metrics(test_labels, test_preds, target_names=class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained image classification model.")
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to the test data directory (must have the same folder structure as train/val)."
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=config.MODEL_SAVE_PATH,
        help=f"Path to the trained model weights file (default: {config.MODEL_SAVE_PATH})."
    )

    args = parser.parse_args()
    print(os.getcwd())
    test(args.data_path, args.model_path)