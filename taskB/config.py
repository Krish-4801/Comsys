# taskB/config.py
import os

# --- PATHS ---
# Assumes a Kaggle-like structure for input data. Change for local execution.
TRAIN_DATA_PATH = "/kaggle/input/comsys-2025/Comys_Hackathon5/Task_B/train"
VAL_DATA_PATH = "/kaggle/input/comsys-2025/Comys_Hackathon5/Task_B/val"

# Output directories (relative to project root)
MODEL_SAVE_DIR = "saved_models"
RESULTS_DIR = "results"
MODEL_NAME = "siamese_model.keras"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

# --- MODEL & DATA PARAMETERS ---
IMG_SIZE = (160, 160)
EMBEDDING_DIM = 128
MAX_TRAIN_SAMPLES = 15000 # Max samples for training pair generation
MAX_TEST_SAMPLES = 4000  # Max samples for validation/testing pair generation

# --- TRAINING PARAMETERS ---
EPOCHS = 12
LEARNING_RATE = 0.001
BATCH_SIZE_PER_REPLICA = 64
SEED = 42