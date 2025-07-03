# src/config.py
import torch

# --- DATA CONFIGURATION ---
# NOTE: Update this path to your actual data directory for training
# For the test script, the path will be passed as a command-line argument.
BASE_DATA_DIR = "/kaggle/input/comsys-new/Comys_Hackathon5/Task_A"
TRAIN_DIR = f"{BASE_DATA_DIR}/train"
VAL_DIR = f"{BASE_DATA_DIR}/val"

# --- MODEL CONFIGURATION ---
MODEL_NAME = "resnet18"
NUM_CLASSES = 2
PRETRAINED = True
MODEL_SAVE_PATH = "saved_models/task_a.pth"

# --- TRAINING CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 32

# Best hyperparameters found from tuning
LEARNING_RATE = 3.28e-05
WEIGHT_DECAY = 6.62e-05

# --- DATA TRANSFORMATION ---
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]