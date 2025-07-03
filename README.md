# COMSYS Hackathon Project: Image Classification & Face Verification

**Team: Pet-a-Byte**

**Contributors:**
*   Aditya Anand
*   Aitijya Ghosh
*   Krish Das

This repository contains the solutions for a two-part computer vision challenge.
*   **Task A:** A binary image classification problem using PyTorch and a ResNet-18 model.
*   **Task B:** A face verification problem using a TensorFlow/Keras-based Siamese Network with a MobileNetV2 backbone.

## Directory Structure

```
└── krish-4801-comsys/
    ├── saved_models/
    │   ├── siamese_model.keras
    │   └── task_a.pth
    ├── taskA/
    │   ├── config.py
    │   ├── data_loader.py
    │   ├── model.py
    │   ├── test.py
    │   ├── train.py
    │   └── utils.py
    └── taskB/
        ├── config.py
        ├── data_loader.py
        ├── model.py
        ├── test.py
        ├── train.py
        └── utils.py
```

## Prerequisites

*   Python 3.8+
*   `pip` for package management

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Krish-4801/Comsys.git
    cd Comsys
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Task A: Image Classification

This task involves a binary classification model built with PyTorch to distinguish between two classes of images (e.g., real vs. spoof).

### Model

The model is a **ResNet-18**, pre-trained on ImageNet, with its final fully connected layer replaced to suit the binary classification task.

### Data Preparation

Before running, you must configure the data paths in `taskA/config.py`. The script expects the data to be organized in the standard `ImageFolder` format:

```
<data_root>/
├── train/
│   ├── class_A/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class_B/
│       ├── image_k.jpg
│       └── ...
└── val/
    ├── class_A/
    │   └── ...
    └── class_B/
        └── ...
```

Update the `BASE_DATA_DIR` variable in `taskA/config.py` to point to your `<data_root>` directory.

### Usage

**Training:**

To train the model from scratch, run the following command from the project root directory (`krish-4801-comsys/`):

```bash
python -m taskA.train
```

The best performing model based on validation accuracy will be saved to `saved_models/task_a.pth`.

**Testing:**

To evaluate the model on a test set, use the `test.py` script. A pre-trained model is provided at `saved_models/task_a.pth`.

```bash
python -m taskA.test --data_path /path/to/your/test_data
```

*   `--data_path`: **(Required)** Path to the test data directory, which must have the same `class_A`/`class_B` structure as the training data.
*   `--model_path`: (Optional) Path to the model weights file. Defaults to `saved_models/task_a.pth`.

### Performance

Results from evaluating the model on the test set. The classes in this specific test were `male` vs. `female`.

--- Test Set Evaluation Results ---
**Accuracy: 92.89%**
**Precision: 93.88%**
**Recall: 96.85%**
**F1 Score: 95.34%**

**📊 Classification Report:**
```
              precision    recall  f1-score   support

      female       0.89      0.81      0.85       105
        male       0.94      0.97      0.95       317

    accuracy                           0.93       422
   macro avg       0.92      0.89      0.90       422
weighted avg       0.93      0.93      0.93       422
```

---

## Task B: Face Verification

This task uses a Siamese Network built with TensorFlow/Keras to determine if two face images belong to the same person.

### Model

The architecture uses two shared **MobileNetV2** backbones to generate 128-dimensional embeddings for each image. The L2 (Euclidean) distance between these embeddings is then calculated and fed into a dense layer with a sigmoid activation to produce a similarity score (1 for match, 0 for non-match).

### Data Preparation

Before running, you must configure the data paths in `taskB/config.py`. The script expects the data to be organized into subdirectories, where each subdirectory name is a unique person ID and contains all images of that person.

```
<data_root>/
├── train/
│   ├── person_id_001/
│   │   ├── img1.jpg
│   │   └── img2.png
│   ├── person_id_002/
│   │   ├── img_a.jpeg
│   │   └── ...
│   └── ...
└── val/
    ├── person_id_001/
    │   └── ...
    └── ...
```

Update the `TRAIN_DATA_PATH` and `VAL_DATA_PATH` variables in `taskB/config.py`.

### Usage

**Training:**

To train the Siamese Network, run the following command from the project root directory:

```bash
python -m taskB.train
```

The script will automatically generate positive and negative pairs, train the model, and save the final model to `saved_models/siamese_model.keras`. A plot of the training history will be saved in the `results/` directory.

**Testing:**

To evaluate the trained Siamese Network on a test set, use the `test.py` script. A pre-trained model is provided at `saved_models/siamese_model.keras`.

```bash
python -m taskB.test --data_path /path/to/your/test_data
```

*   `--data_path`: **(Required)** Path to the test data directory, which must be structured with person-specific subdirectories.
*   `--model_path`: (Optional) Path to the saved Keras model file. Defaults to `saved_models/siamese_model.keras`.
*   `--max_samples`: (Optional) Maximum number of pairs to generate for testing.
*   `--batch_size`: (Optional) Batch size for prediction.

### Performance

Results from evaluating the Siamese network on 4000 pairs (2000 matching, 2000 non-matching) from the test set.

**--- Evaluation Metrics ---**
*   ✅ **Accuracy:**  0.7795
*   🎯 **Precision:** 0.9027
*   🔄 **Recall:**    0.6265
*   📈 **F1-Score:**  0.7397

**--- Classification Report ---**
```
               precision    recall  f1-score   support

Non-Match (0)       0.71      0.93      0.81      2000
    Match (1)       0.90      0.63      0.74      2000

     accuracy                           0.78      4000
    macro avg       0.81      0.78      0.77      4000
 weighted avg       0.81      0.78      0.77      4000
```

